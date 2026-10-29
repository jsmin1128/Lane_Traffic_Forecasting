"""
날짜별 정규화 텐서(플래그 포함)로 학습용 슬라이딩 윈도우를 만듭니다.

  입력: <tensor_dir>/<날짜>/normalized_tensor_<날짜>_with_flag.npy  (288, 1370, 9)
  X:    과거 12스텝(1시간) (12, 1370, 9) = 대기 차량 수 4 + 평균 속도 4 + 주말·공휴일 플래그 1
  Y:    다음 1스텝(5분)의 8채널 (1370, 8)
  출력: <out_dir>/all_X.npy, all_Y.npy, all_DATE.npy (윈도우별 날짜)
"""

import os

import numpy as np

from traffic import config

INPUT_CHANNELS  = config.NUM_TRAFFIC + 1   # queue+spd 8 + 플래그 1
OUTPUT_CHANNELS = config.NUM_TRAFFIC       # 예측 대상 채널 수 (queue+spd 8개)


def make_windows(tensor_dir, out_dir, dates):
    NUM_SLOTS, WINDOW_SIZE = config.NUM_SLOTS, config.WINDOW_SIZE
    os.makedirs(out_dir, exist_ok=True)

    # ┌──────────────────────────────────────────────────────────────────────┐
    # │ 1. 슬라이딩 윈도우 생성                                                │
    # └──────────────────────────────────────────────────────────────────────┘
    all_X = []      # List of input windows (각 원소 shape=(12,1370,9))
    all_Y = []      # List of target windows (각 원소 shape=(1370,8))
    all_DATE = []   # 윈도우별 DATE 메타정보

    for DATE in dates:
        tensor_path = os.path.join(tensor_dir, str(DATE), f'normalized_tensor_{DATE}_with_flag.npy')

        # 파일 유무 확인
        if not os.path.exists(tensor_path):
            print(f"⚠️ Skipping {DATE}: 파일이 없습니다 ({tensor_path})")
            continue

        # (1) 정규화 + 플래그가 붙은 텐서 로드
        daily = np.load(tensor_path)

        # (2) 텐서 shape 검증 (예상: (288,1370,9))
        if daily.shape[0] != NUM_SLOTS or daily.shape[2] != INPUT_CHANNELS:
            raise ValueError(f"[{DATE}] 텐서 shape이 예상과 다릅니다: {daily.shape} (예상: (288,1370,9))")

        # (3) 슬라이딩 윈도우: start = 0 ~ (288-12-1) = 275까지
        #     - X 윈도우: daily[start : start+12, :, :] → shape=(12,1370,9)
        #     - Y 타깃:  daily[start+12, :, :8]       → shape=(1370,8)
        for start in range(NUM_SLOTS - WINDOW_SIZE):
            x_win = daily[start : start + WINDOW_SIZE]               # (12,1370,9)
            y_next = daily[start + WINDOW_SIZE, :, :OUTPUT_CHANNELS]  # (1370,8)

            all_X.append(x_win)
            all_Y.append(y_next)
            all_DATE.append(DATE)

    # (4) 리스트 → NumPy 배열로 변환
    all_X = np.stack(all_X, axis=0)
    all_Y = np.stack(all_Y, axis=0)
    all_DATE = np.array(all_DATE)         # shape=(총윈도우,)

    # ┌──────────────────────────────────────────────────────────────────────┐
    # │ 2. 검증                                                               │
    # └──────────────────────────────────────────────────────────────────────┘
    print("\n=== 슬라이딩 윈도우 생성 후 검증 ===")
    print(f"▶ 총 윈도우 개수: {all_X.shape[0]}")
    print(f"▶ all_X shape: {all_X.shape}  (윈도우 수, 12, 1370, 9)")
    print(f"▶ all_Y shape: {all_Y.shape}  (윈도우 수, 1370, 8)")
    print("▶ all_X dtype:", all_X.dtype)
    print("▶ all_Y dtype:", all_Y.dtype)
    assert all_X.dtype in [np.float32, np.float64], "all_X는 실수형이어야 합니다."
    assert all_Y.dtype in [np.float32, np.float64], "all_Y는 실수형이어야 합니다."

    # NaN/Inf 검사
    assert not np.isnan(all_X).any() and not np.isinf(all_X).any(), "all_X에 NaN 또는 Inf가 존재합니다."
    assert not np.isnan(all_Y).any() and not np.isinf(all_Y).any(), "all_Y에 NaN 또는 Inf가 존재합니다."
    print("▶ NaN/Inf 없음")

    # 연속성 검증: 첫 윈도우의 Y(다음 슬롯)는 같은 날 두 번째 윈도우 X의 마지막 스텝과 같아야 함
    assert np.array_equal(all_Y[0], all_X[1, -1, :, :OUTPUT_CHANNELS]), \
        "연속성 검증 실패: Y[0]이 X[1]의 마지막 스텝과 일치하지 않습니다."
    print("▶ 연속성 검증 통과: Y[0] == X[1]의 마지막 스텝")

    # ┌──────────────────────────────────────────────────────────────────────┐
    # │ 3. 결과물 저장                                                         │
    # └──────────────────────────────────────────────────────────────────────┘
    np.save(os.path.join(out_dir, 'all_X.npy'), all_X)
    np.save(os.path.join(out_dir, 'all_Y.npy'), all_Y)
    np.save(os.path.join(out_dir, 'all_DATE.npy'), all_DATE)
    print(f"\n✅ 슬라이딩 윈도우 저장 완료 → {out_dir}")
