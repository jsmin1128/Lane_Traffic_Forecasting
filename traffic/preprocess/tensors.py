"""
od.parquet → 날짜별 텐서 (<tensor_dir>/<날짜>/)

build_tensors
  input_tensor_<날짜>.npy        (288 슬롯, N 차로, 8채널) 실제 값
  normalized_tensor_<날짜>.npy   위 텐서를 날짜·채널별 z-score로 정규화한 것

    채널 0~3  차종별 대기 차량 수: 5분 동안 그 차로에서 대기행렬에 잡힌(que_all=1) 서로 다른 차량 수
    채널 4~7  차종별 평균 속도(m/s): 속도 > 0인 기록의 평균. 움직이는 차 기록이 없으면 0
    차종 순서는 승용, 버스, 트럭(대형트럭 포함), 오토바이이고 특수차량은 뺍니다.
    lanes.txt에 없는 차로의 기록은 버립니다.

add_flags
  normalized_tensor_<날짜>_with_flag.npy   정규화 텐서 뒤에 주말·공휴일 플래그 채널을 붙인 것 (9채널)
"""

from datetime import datetime
from pathlib import Path

import holidays
import numpy as np
import pandas as pd

from traffic import config

# 차종 코드(vhcl_typ) → 차종 번호. 여기 없는 특수차량(5)은 버립니다.
CATEGORY = {1: 0,   # 승용차
            2: 1,   # 버스
            3: 2,   # 트럭
            4: 2,   # 대형트럭 → 트럭
            6: 3}   # 오토바이
NUM_CATEGORIES = 4


def read_od(od_parquet, date):
    """하루치 od 기록(13개 route 전부)을 읽고 5분 슬롯과 차종 번호를 붙입니다."""
    od = pd.read_parquet(od_parquet, filters=[('date', '=', date)],
                         columns=['unix_time', 'vhcl_id', 'vhcl_typ', 'lane_id', 'que_all', 'spd'])
    kst = pd.to_datetime(od['unix_time'], unit='s', utc=True).dt.tz_convert('Asia/Seoul')
    od['slot'] = (kst.dt.hour * 60 + kst.dt.minute) // 5    # 하루 중 5분 슬롯 0~287
    od['cat'] = od['vhcl_typ'].map(CATEGORY)
    return od.dropna(subset=['cat']).astype({'cat': int})


def date_tensor(od, lane_ids):
    """하루치 기록 → (288, N, 8) 텐서"""
    keys = ['slot', 'lane_id', 'cat']
    queue = od[od['que_all'] == 1].groupby(keys)['vhcl_id'].nunique()   # 대기행렬에 잡힌 서로 다른 차량 수
    speed = od[od['spd'] > 0].groupby(keys)['spd'].mean()              # 움직이는 차 기록의 평균 속도

    x = np.zeros((config.NUM_SLOTS, len(lane_ids), 2 * NUM_CATEGORIES), dtype=np.float32)
    lane_index = {lane: i for i, lane in enumerate(lane_ids)}
    for values, first_channel in [(queue, 0), (speed, NUM_CATEGORIES)]:   # 대기 → 채널 0~3, 속도 → 4~7
        df = values.rename('value').reset_index()
        df['lane'] = df['lane_id'].map(lane_index)
        df = df[df['lane'].notna()]                                    # lanes.txt에 없는 차로는 버림
        slot, lane, cat = df['slot'].to_numpy(), df['lane'].to_numpy(int), df['cat'].to_numpy()
        x[slot, lane, first_channel + cat] = df['value'].to_numpy(float)
    return x


def normalize(x):
    """채널별 z-score. 값이 모두 같은 채널(표준편차 0)은 0으로 둡니다."""
    out = np.zeros_like(x)
    for ch in range(x.shape[-1]):
        values = x[..., ch].ravel()
        mean, std = values.mean(), values.std()
        if std > 0:
            out[..., ch] = (x[..., ch] - mean) / std
    return out


def build_tensors(od_parquet, lanes_path, tensor_dir, dates):
    """dates마다 input_tensor, normalized_tensor를 저장합니다."""
    lane_ids = Path(lanes_path).read_text(encoding='utf-8').split()
    for date in dates:
        od = read_od(od_parquet, date)
        x = date_tensor(od, lane_ids)

        date_dir = Path(tensor_dir) / str(date)
        date_dir.mkdir(parents=True, exist_ok=True)
        np.save(date_dir / f'input_tensor_{date}.npy', x)
        np.save(date_dir / f'normalized_tensor_{date}.npy', normalize(x))
        print(f"▶ {date}: 기록 {len(od):,}건 → {x.shape}")


KR_HOLIDAYS = holidays.KR()


def is_day_off(date):
    """주말이거나 한국 공휴일이면 True"""
    day = datetime.strptime(str(date), '%Y%m%d').date()
    return day.weekday() >= 5 or day in KR_HOLIDAYS


def add_flags(tensor_dir, dates):
    """normalized_tensor 뒤에 플래그 채널(그날 전부 1 또는 0)을 붙입니다 (8 → 9채널)."""
    for date in dates:
        date_dir = Path(tensor_dir) / str(date)
        x = np.load(date_dir / f'normalized_tensor_{date}.npy')
        flag = np.full((*x.shape[:2], 1), 1.0 if is_day_off(date) else 0.0, dtype=np.float32)
        np.save(date_dir / f'normalized_tensor_{date}_with_flag.npy', np.concatenate([x, flag], axis=2))
    days_off = [d for d in dates if is_day_off(d)]
    print(f"▶ 주말·공휴일 플래그: {len(dates)}일 중 {len(days_off)}일 → {days_off}")
