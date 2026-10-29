"""
od.parquet → 날짜별 텐서 → <out_dir>/values.npy, normalized.npy, dates.npy

  values.npy      (날짜, 288 슬롯, N 차로, 8채널) 실제 값
  normalized.npy  values를 날짜·채널별 z-score로 정규화한 것 (windows 단계의 입력)
  dates.npy       (날짜,) 위 두 배열의 첫 번째 축 순서

  채널 0~3  차종별 대기 차량 수: 5분 동안 그 차로에서 대기행렬에 잡힌(que_all=1) 서로 다른 차량 수
  채널 4~7  차종별 평균 속도(m/s): 속도 > 0인 기록의 평균. 움직이는 차 기록이 없으면 0
  차종 순서는 승용, 버스, 트럭(대형트럭 포함), 오토바이이고 특수차량은 뺍니다.
  lanes.txt에 없는 차로의 기록은 버립니다.
"""

from pathlib import Path

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

    x = np.zeros((config.NUM_SLOTS, len(lane_ids), config.NUM_CHANNELS), dtype=np.float32)
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


def build_tensors(od_parquet, lanes_path, out_dir, dates):
    """dates의 하루치 텐서를 순서대로 쌓아 저장합니다."""
    lane_ids = Path(lanes_path).read_text(encoding='utf-8').split()
    values = np.zeros((len(dates), config.NUM_SLOTS, len(lane_ids), config.NUM_CHANNELS), dtype=np.float32)
    normalized = np.zeros_like(values)
    for i, date in enumerate(dates):
        od = read_od(od_parquet, date)
        values[i] = date_tensor(od, lane_ids)
        normalized[i] = normalize(values[i])
        print(f"▶ {date}: 기록 {len(od):,}건")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / 'values.npy', values)
    np.save(out_dir / 'normalized.npy', normalized)
    np.save(out_dir / 'dates.npy', np.array(dates))
    print(f"▶ 텐서 {values.shape} → {out_dir}/values.npy, normalized.npy, dates.npy")
