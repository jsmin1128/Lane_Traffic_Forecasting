#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
check_windows.py

processed/windows의 X.npy, Y.npy, dates.npy가 올바른지 확인합니다.

검증 내용:
1) 파일이 있는지
2) shape, dtype, NaN/Inf, 플래그 채널이 그날의 주말·공휴일 여부와 같은지
3) 연속성: 날짜마다 첫 윈도우의 Y(다음 슬롯)가 같은 날 두 번째 윈도우 X의 마지막 스텝과 같은지

사용법: python scripts/check/check_windows.py
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from traffic import config
from traffic.preprocess.windows import is_day_off

WINDOWS_DIR     = config.PROCESSED_DIR / 'windows'
INPUT_CHANNELS  = config.NUM_CHANNELS + 1   # 8채널 + 주말·공휴일 플래그
OUTPUT_CHANNELS = config.NUM_CHANNELS


def main():
    print("\n=== 1) 파일 확인 ===")
    paths = {name: WINDOWS_DIR / f'{name}.npy' for name in ['X', 'Y', 'dates']}
    for name, path in paths.items():
        if not path.exists():
            sys.exit(f"{name}.npy 파일이 없습니다: {path}")
        print(f"✔ {path}")

    print("\n=== 2) shape·dtype·NaN/Inf·플래그 검증 ===")
    X = np.load(paths['X'], mmap_mode='r')   # 약 5GB라 메모리에 한 번에 올리지 않음
    Y = np.load(paths['Y'], mmap_mode='r')
    dates = np.load(paths['dates'])
    print(f"▶ X {X.shape} {X.dtype} | Y {Y.shape} {Y.dtype} | dates {dates.shape}")

    n, num_nodes = X.shape[0], X.shape[2]
    assert Y.shape[0] == n and dates.shape[0] == n, "윈도우 개수가 서로 다릅니다."
    assert X.shape[1:] == (config.WINDOW_SIZE, num_nodes, INPUT_CHANNELS), f"X shape 오류: {X.shape}"
    assert Y.shape[1:] == (num_nodes, OUTPUT_CHANNELS), f"Y shape 오류: {Y.shape}"
    assert X.dtype == np.float32 and Y.dtype == np.float32, "float32여야 합니다."
    flag = np.array([1.0 if is_day_off(d) else 0.0 for d in dates], dtype=np.float32)   # 윈도우별 기대 플래그
    for start in range(0, n, 500):
        x, y = np.asarray(X[start:start + 500]), np.asarray(Y[start:start + 500])
        assert np.isfinite(x).all() and np.isfinite(y).all(), f"NaN 또는 Inf가 있습니다 (윈도우 {start}~)."
        assert (x[..., -1] == flag[start:start + 500, None, None]).all(), \
            f"플래그 채널이 주말·공휴일 여부와 다릅니다 (윈도우 {start}~)."
    days, counts = np.unique(dates, return_counts=True)
    days_off = sum(is_day_off(d) for d in days)
    print(f"▶ 윈도우 {n}개, 날짜 {len(days)}개, 날짜별 윈도우 수 {sorted(set(counts.tolist()))}, 주말·공휴일 {days_off}일")
    print("▶ shape·dtype·NaN/Inf·플래그 검증 통과 ✓")

    print("\n=== 3) 연속성 검증 (날짜마다 첫 두 윈도우) ===")
    firsts = np.flatnonzero(np.r_[True, dates[1:] != dates[:-1]])   # 날짜별 첫 윈도우 위치
    for i in firsts:
        assert dates[i + 1] == dates[i], f"{dates[i]}: 윈도우가 1개뿐입니다."
        assert np.array_equal(Y[i], X[i + 1, -1, :, :OUTPUT_CHANNELS]), \
            f"{dates[i]}: Y[{i}]가 X[{i + 1}]의 마지막 스텝과 일치하지 않습니다."
    print(f"▶ {len(firsts)}개 날짜 모두 연속성 검증 통과 ✓")

    print("\n✅ 저장된 윈도우 검증 완료 ✓")


if __name__ == '__main__':
    main()
