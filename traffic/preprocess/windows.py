"""
날짜별 정규화 텐서 → 학습 샘플(슬라이딩 윈도우) → <out_dir>/X.npy, Y.npy, dates.npy

  X.npy      (윈도우 수, 12, N, 9)  과거 12스텝(1시간): 8채널 + 주말·공휴일 플래그
  Y.npy      (윈도우 수, N, 8)      바로 다음 5분의 8채널
  dates.npy  (윈도우 수,)           윈도우의 날짜

입력은 tensors/normalized.npy와 tensors/dates.npy입니다.
플래그 채널은 그날이 주말이거나 한국 공휴일이면 전부 1, 아니면 0입니다.
윈도우는 하루 안에서만 5분씩 밀며 만들어 하루 288 - 12 = 276개입니다.
첫 정답이 01:00 슬롯이라 00:00~00:55는 예측 대상이 아닙니다.
"""

from datetime import datetime
from pathlib import Path

import holidays
import numpy as np

from traffic import config

KR_HOLIDAYS = holidays.KR()


def is_day_off(date):
    """주말이거나 한국 공휴일이면 True"""
    day = datetime.strptime(str(date), '%Y%m%d').date()
    return day.weekday() >= 5 or day in KR_HOLIDAYS


def build_windows(tensor_dir, out_dir):
    normalized = np.load(Path(tensor_dir) / 'normalized.npy')    # (날짜, 288, N, 8)
    dates = np.load(Path(tensor_dir) / 'dates.npy')
    T = config.WINDOW_SIZE

    X, Y, D = [], [], []
    for day, date in zip(normalized, dates):
        flag = np.full((*day.shape[:2], 1), 1.0 if is_day_off(date) else 0.0, dtype=np.float32)
        day = np.concatenate([day, flag], axis=2)                 # (288, N, 9)
        for start in range(config.NUM_SLOTS - T):
            X.append(day[start:start + T])                        # 입력: 슬롯 start ~ start+11
            Y.append(day[start + T, :, :config.NUM_CHANNELS])     # 정답: 바로 다음 슬롯 start+12
            D.append(date)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / 'X.npy', np.stack(X))
    np.save(out_dir / 'Y.npy', np.stack(Y))
    np.save(out_dir / 'dates.npy', np.array(D))
    days_off = [int(d) for d in dates if is_day_off(d)]
    print(f"▶ 윈도우 {len(X):,}개 (날짜 {len(dates)}개 × {config.NUM_SLOTS - T}), "
          f"주말·공휴일 {len(days_off)}일 {days_off} → {out_dir}/X.npy, Y.npy, dates.npy")
