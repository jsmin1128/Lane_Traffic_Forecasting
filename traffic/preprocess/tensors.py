"""
날짜별 텐서를 만듭니다.

build_tensors: od.parquet → <out_dir>/<날짜>/input_tensor_<날짜>.npy, normalized_tensor_<날짜>.npy
    shape = (288 슬롯, N 차로, 8 채널) = 차종 4종 대기 차량 수 + 차종 4종 평균 속도
    정규화는 날짜·채널별 Z-score입니다.

add_flags: 위 두 텐서 뒤에 "주말 또는 공휴일" 플래그 채널을 붙여
    input_tensor_<날짜>_with_flag.npy, normalized_tensor_<날짜>_with_flag.npy (9채널)로 저장합니다.
"""

import os
import warnings

import holidays
import numpy as np
import pandas as pd

from traffic import config

NUM_CATEGORIES = 4                   # 승용, 버스, 트럭, 오토바이
CHANNELS       = NUM_CATEGORIES * 2  # queue+spd → 8

# 차량 종류(vhcl_typ) → 카테고리 맵핑
VHC_TYP_MAPPING = {
    1: 0,  # 승용차 → 0
    2: 1,  # 버스   → 1
    3: 2,  # 트럭   → 2
    4: 2,  # 대형트럭→ 트럭(2)
    5: None,  # 특수차량 → 제거
    6: 3   # 오토바이 → 3
}


def build_date_tensor(date: int, od_parquet, lane_ids, out_dir):
    """
    주어진 날짜(date, YYYYMMDD)별로:
      1) Parquet에서 해당 날짜 데이터 로드
      2) 전처리 → 입력 텐서(input_tensor) 생성
      3) Z-Score 정규화 → normalized_tensor 생성
      4) input_tensor 및 normalized_tensor를 .npy로 저장
    """
    NUM_SLOTS = config.NUM_SLOTS
    NUM_LANES = len(lane_ids)
    date_dir = os.path.join(out_dir, str(date))
    os.makedirs(date_dir, exist_ok=True)

    print(f"\n========== Processing Date: {date} ==========")
    # ──────────────────────────────────────────────────────────────────────────
    # 1) Parquet에서 해당 날짜 데이터 로드
    # ──────────────────────────────────────────────────────────────────────────
    print("1) Reading Parquet for date =", date)
    try:
        od_df = pd.read_parquet(
            str(od_parquet),
            engine='pyarrow',
            filters=[('date', '=', date)]
        )
    except Exception as e:
        print(f"  ▶ Error: {date} 관련 Parquet 로드 실패: {e}")
        return

    # 검증: date 컬럼이 모두 동일한지 확인
    unique_dates = od_df['date'].astype(int).unique()
    if not (len(unique_dates) == 1 and unique_dates[0] == date):
        print(f"  ▶ Warning: date 필터링 결과가 예상과 다릅니다: {unique_dates}")
    print("  ▶ od_df shape:", od_df.shape)

    # ──────────────────────────────────────────────────────────────────────────
    # 2) 한국시간(dt_kst) 생성 및 원본 dt 열 삭제
    # ──────────────────────────────────────────────────────────────────────────
    print("2) Generating dt_kst (KST)")
    od_df['dt_utc'] = pd.to_datetime(od_df['unix_time'], unit='s', utc=True)
    od_df['dt_kst'] = od_df['dt_utc'].dt.tz_convert('Asia/Seoul')
    if 'dt' in od_df.columns:
        od_df.drop(columns=['dt'], inplace=True)

    # ──────────────────────────────────────────────────────────────────────────
    # 3) 차량 종류(vhcl_typ) 재분류 및 특수차량(5) 제거
    # ──────────────────────────────────────────────────────────────────────────
    print("3) Mapping vhcl_typ → vhcl_cat and dropping special vehicles")
    od_df = od_df[od_df['vhcl_typ'] != 5].copy()  # 특수차량 제거
    od_df['vhcl_cat'] = od_df['vhcl_typ'].map(VHC_TYP_MAPPING)
    od_df.dropna(subset=['vhcl_cat'], inplace=True)
    od_df['vhcl_cat'] = od_df['vhcl_cat'].astype(int)

    # ──────────────────────────────────────────────────────────────────────────
    # 4) 5분 단위 슬롯(slot) 생성: 0 ~ 287
    # ──────────────────────────────────────────────────────────────────────────
    print("4) Generating 5-minute slot")
    od_df['slot'] = (
        (od_df['dt_kst'].dt.hour * 60 + od_df['dt_kst'].dt.minute) // 5
    ).astype(int)

    # ──────────────────────────────────────────────────────────────────────────
    # 5) Queue Count 계산 (que_all == 1인 경우)
    # ──────────────────────────────────────────────────────────────────────────
    print("5) Calculating queue_count")
    queue_df = od_df[od_df['que_all'] == 1]
    queue_count = (
        queue_df
        .groupby(['slot', 'lane_id', 'vhcl_cat'])['vhcl_id']
        .nunique()
        .reset_index(name='queue_count')
    )

    # ──────────────────────────────────────────────────────────────────────────
    # 6) 평균 속도(avg_spd) 계산 (spd > 0인 경우)
    # ──────────────────────────────────────────────────────────────────────────
    print("6) Calculating avg_spd")
    moving_df = od_df[od_df['spd'] > 0]
    spd_avg = (
        moving_df
        .groupby(['slot', 'lane_id', 'vhcl_cat'])['spd']
        .mean()
        .reset_index(name='avg_spd')
    )

    # ──────────────────────────────────────────────────────────────────────────
    # 7) 입력 텐서(input_tensor) 초기화
    # ──────────────────────────────────────────────────────────────────────────
    print("7) Initializing input_tensor array")
    input_tensor = np.zeros(
        (NUM_SLOTS, NUM_LANES, CHANNELS),
        dtype=np.float32
    )

    # ──────────────────────────────────────────────────────────────────────────
    # 8) queue_count 데이터를 input_tensor 채널(0~3)에 배치
    # ──────────────────────────────────────────────────────────────────────────
    print("8) Filling queue channels (0~3)")
    for _, row in queue_count.iterrows():
        slot = int(row['slot'])
        lane = row['lane_id']
        cat = int(row['vhcl_cat'])
        cnt = int(row['queue_count'])
        if lane in lane_ids:
            lane_idx = lane_ids.index(lane)
            input_tensor[slot, lane_idx, cat] = cnt

    # ──────────────────────────────────────────────────────────────────────────
    # 9) avg_spd 데이터를 input_tensor 채널(4~7)에 배치
    # ──────────────────────────────────────────────────────────────────────────
    print("9) Filling speed channels (4~7)")
    for _, row in spd_avg.iterrows():
        slot = int(row['slot'])
        lane = row['lane_id']
        cat = int(row['vhcl_cat'])
        spd_val = float(row['avg_spd'])
        if lane in lane_ids:
            lane_idx = lane_ids.index(lane)
            input_tensor[slot, lane_idx, NUM_CATEGORIES + cat] = spd_val

    # ──────────────────────────────────────────────────────────────────────────
    # 10) input_tensor NaN/음수/Shape 검증
    # ──────────────────────────────────────────────────────────────────────────
    print("10) Validating input_tensor")
    if np.isnan(input_tensor).any():
        raise ValueError(f"[{date}] input_tensor에 NaN이 존재합니다.")
    if (input_tensor < 0).any():
        raise ValueError(f"[{date}] input_tensor에 음수 값이 존재합니다.")
    if input_tensor.shape != (NUM_SLOTS, NUM_LANES, CHANNELS):
        raise ValueError(f"[{date}] input_tensor shape 오류: 현재 {input_tensor.shape}")

    print("    ✔ input_tensor validation passed")

    # ──────────────────────────────────────────────────────────────────────────
    # 11) input_tensor 저장 (.npy)
    # ──────────────────────────────────────────────────────────────────────────
    print("11) Saving input_tensor")
    input_npy_path = os.path.join(date_dir, f'input_tensor_{date}.npy')
    np.save(input_npy_path, input_tensor)
    print(f"    ▶ Saved: {input_npy_path}")

    # ──────────────────────────────────────────────────────────────────────────
    # 12) 채널별 통계(평균·표준편차) 계산
    # ──────────────────────────────────────────────────────────────────────────
    print("12) Calculating channel-wise mean & std for normalization")
    means = []
    stds  = []
    for ch in range(CHANNELS):
        vals = input_tensor[:, :, ch].ravel()
        mu    = np.mean(vals)
        sigma = np.std(vals)
        means.append(mu)
        stds.append(sigma)

    means = np.array(means, dtype=np.float32)
    stds  = np.array(stds, dtype=np.float32)

    # ──────────────────────────────────────────────────────────────────────────
    # 13) Z-Score 정규화 수행 → normalized_tensor 생성
    # ──────────────────────────────────────────────────────────────────────────
    print("13) Performing Z-Score normalization")
    normalized_tensor = np.zeros_like(input_tensor, dtype=np.float32)
    for ch in range(CHANNELS):
        mu    = means[ch]
        sigma = stds[ch]
        if sigma == 0:
            normalized_tensor[:, :, ch] = 0.0
        else:
            normalized_tensor[:, :, ch] = (input_tensor[:, :, ch] - mu) / sigma

    # 정규화 후 검증 (평균 0, 표준편차 ≈ 1)
    for ch in range(CHANNELS):
        vals_norm = normalized_tensor[:, :, ch].ravel()
        mean_ch   = np.mean(vals_norm)
        std_ch    = np.std(vals_norm)
        if not np.isclose(mean_ch, 0.0, atol=1e-4) or not np.isclose(std_ch, 1.0, atol=1e-4):
            print(f"    ▶ Warning: 채널 {ch} 정규화 통계 이상 (mean={mean_ch:.4f}, std={std_ch:.4f})")

    # ──────────────────────────────────────────────────────────────────────────
    # 14) normalized_tensor 저장 (.npy)
    # ──────────────────────────────────────────────────────────────────────────
    print("14) Saving normalized_tensor")
    norm_npy_path = os.path.join(date_dir, f'normalized_tensor_{date}.npy')
    np.save(norm_npy_path, normalized_tensor)
    print(f"    ▶ Saved: {norm_npy_path}")

    print(f"========== Completed Date: {date} ==========\n")


def build_tensors(od_parquet, lanes_path, out_dir, dates):
    """dates의 모든 날짜에 대해 build_date_tensor를 순서대로 실행합니다."""
    # lane_ids 로드 (한 줄당 하나의 lane_id)
    with open(lanes_path, 'r', encoding='utf-8') as f:
        lane_ids = [line.strip() for line in f.readlines()]

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')   # pandas 경고 감추기
        for date in dates:
            build_date_tensor(date, od_parquet, lane_ids, out_dir)

    print(f"\nAll dates processed. Tensors saved in per-date folders under {out_dir}/")


def add_flags(tensor_dir, dates):
    """input_tensor·normalized_tensor(.npy)에 "주말 또는 공휴일" 플래그 채널을 추가합니다 (8 → 9채널)."""
    # 한국 공휴일 캘린더 (연도 자동 확장)
    kr_holidays = holidays.KR()

    for DATE in dates:
        date_dir = os.path.join(tensor_dir, str(DATE))

        input_path = os.path.join(date_dir, f'input_tensor_{DATE}.npy')
        norm_path  = os.path.join(date_dir, f'normalized_tensor_{DATE}.npy')
        if not os.path.exists(input_path) or not os.path.exists(norm_path):
            print(f"⚠️ Skipping {DATE}: 파일 누락")
            continue

        # 1) 텐서 로드
        input_tensor      = np.load(input_path)   # (288,1370,8)
        normalized_tensor = np.load(norm_path)    # (288,1370,8)

        # 2) DATE의 요일 및 공휴일 여부 계산
        ts = pd.to_datetime(str(DATE), format='%Y%m%d')
        weekday_idx = ts.weekday()                 # 0=월, …, 5=토,6=일
        is_weekend  = (weekday_idx >= 5)
        is_holiday  = (ts.date() in kr_holidays)

        # 주말 또는 공휴일 플래그
        weekend_hol_flag = 1.0 if (is_weekend or is_holiday) else 0.0

        # 슬롯(288) × 노드(1370)로 브로드캐스트 → (288,1370,1)
        num_slots = input_tensor.shape[0]
        num_lanes = input_tensor.shape[1]
        feat = np.full((num_slots, num_lanes, 1), weekend_hol_flag, dtype=np.float32)

        # 3) 기존 텐서 뒤에 붙이기
        aug_input = np.concatenate([input_tensor,      feat], axis=2)
        aug_norm  = np.concatenate([normalized_tensor, feat], axis=2)

        # 4) 저장
        out_input_path = os.path.join(date_dir, f'input_tensor_{DATE}_with_flag.npy')
        out_norm_path  = os.path.join(date_dir, f'normalized_tensor_{DATE}_with_flag.npy')
        np.save(out_input_path, aug_input)
        np.save(out_norm_path,  aug_norm)

        print(f"● {DATE} → shapes: {aug_input.shape}, flag={weekend_hol_flag}")

    print("▶ 모든 날짜 처리 완료")
