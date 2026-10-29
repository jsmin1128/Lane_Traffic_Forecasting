"""
날짜별 텐서 → 학습 샘플(슬라이딩 윈도우) → <out_dir>/all_X.npy, all_Y.npy, all_DATE.npy

  all_X     (윈도우 수, 12, N, 9)   과거 12스텝(1시간): 대기 4 + 속도 4 + 주말·공휴일 플래그 1
  all_Y     (윈도우 수, N, 8)       바로 다음 5분의 8채널 (플래그 제외)
  all_DATE  (윈도우 수,)            윈도우의 날짜

입력은 날짜별 normalized_tensor_<날짜>_with_flag.npy (288, N, 9)입니다.
윈도우는 하루 안에서만 5분씩 밀며 만들어 하루 288 - 12 = 276개입니다.
첫 정답이 01:00 슬롯이라 00:00~00:55는 예측 대상이 아닙니다.
"""

from pathlib import Path

import numpy as np

from traffic import config


def make_windows(tensor_dir, out_dir, dates):
    T = config.WINDOW_SIZE
    X, Y, D = [], [], []
    for date in dates:
        day = np.load(Path(tensor_dir) / str(date) / f'normalized_tensor_{date}_with_flag.npy')
        assert day.shape[0] == config.NUM_SLOTS and day.shape[2] == config.NUM_TRAFFIC + 1, f"{date}: {day.shape}"
        for start in range(config.NUM_SLOTS - T):
            X.append(day[start:start + T])                        # 입력: 슬롯 start ~ start+11
            Y.append(day[start + T, :, :config.NUM_TRAFFIC])      # 정답: 바로 다음 슬롯 start+12
            D.append(date)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / 'all_X.npy', np.stack(X))
    np.save(out_dir / 'all_Y.npy', np.stack(Y))
    np.save(out_dir / 'all_DATE.npy', np.array(D))
    print(f"▶ 윈도우 {len(X):,}개 (날짜 {len(dates)}개 × {config.NUM_SLOTS - T}) → {out_dir}/all_X.npy, all_Y.npy, all_DATE.npy")
