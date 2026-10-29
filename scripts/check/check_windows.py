#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
check_windows.py

processed/windows의 all_X.npy, all_Y.npy, all_DATE.npy가 올바른지 확인합니다.

검증 내용:
1) 파일이 존재하는지
2) 배열 shape, dtype, NaN/Inf 유무
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

WINDOWS_DIR     = config.PROCESSED_DIR / 'windows'
INPUT_CHANNELS  = config.NUM_TRAFFIC + 1   # queue+spd 8 + 주말·공휴일 플래그 1
OUTPUT_CHANNELS = config.NUM_TRAFFIC


def main():
    print("\n=== 1) 파일 존재 여부 확인 ===")
    paths = {name: WINDOWS_DIR / f'{name}.npy' for name in ['all_X', 'all_Y', 'all_DATE']}
    for name, path in paths.items():
        if not path.exists():
            sys.exit(f"{name}.npy 파일이 없습니다: {path}")
        print(f"✔ {name}.npy ({path})")

    print("\n=== 2) shape·dtype·NaN/Inf 검증 ===")
    all_X = np.load(paths['all_X'], mmap_mode='r')   # 약 5GB라 메모리에 한 번에 올리지 않음
    all_Y = np.load(paths['all_Y'], mmap_mode='r')
    all_DATE = np.load(paths['all_DATE'])
    print(f"▶ all_X {all_X.shape} {all_X.dtype} | all_Y {all_Y.shape} {all_Y.dtype} | all_DATE {all_DATE.shape}")

    n, num_nodes = all_X.shape[0], all_X.shape[2]
    assert all_Y.shape[0] == n and all_DATE.shape[0] == n, "윈도우 개수가 서로 다릅니다."
    assert all_X.shape[1:] == (config.WINDOW_SIZE, num_nodes, INPUT_CHANNELS), f"all_X shape 오류: {all_X.shape}"
    assert all_Y.shape[1:] == (num_nodes, OUTPUT_CHANNELS), f"all_Y shape 오류: {all_Y.shape}"
    assert all_X.dtype in [np.float32, np.float64] and all_Y.dtype in [np.float32, np.float64], "실수형이어야 합니다."
    for start in range(0, n, 500):
        for name, arr in (('all_X', all_X), ('all_Y', all_Y)):
            chunk = np.asarray(arr[start:start + 500])
            assert np.isfinite(chunk).all(), f"{name}에 NaN 또는 Inf가 있습니다 (윈도우 {start}~)."
    dates, counts = np.unique(all_DATE, return_counts=True)
    print(f"▶ 윈도우 {n}개, 날짜 {len(dates)}개, 날짜별 윈도우 수 {sorted(set(counts.tolist()))}")
    print("▶ shape·dtype·NaN/Inf 검증 통과 ✓")

    print("\n=== 3) 연속성 검증 (날짜마다 첫 두 윈도우) ===")
    firsts = np.flatnonzero(np.r_[True, all_DATE[1:] != all_DATE[:-1]])   # 날짜별 첫 윈도우 위치
    for i in firsts:
        assert all_DATE[i + 1] == all_DATE[i], f"{all_DATE[i]}: 윈도우가 1개뿐입니다."
        assert np.array_equal(all_Y[i], all_X[i + 1, -1, :, :OUTPUT_CHANNELS]), \
            f"{all_DATE[i]}: Y[{i}]가 X[{i + 1}]의 마지막 스텝과 일치하지 않습니다."
    print(f"▶ {len(firsts)}개 날짜 모두 연속성 검증 통과 ✓")

    print("\n✅ 저장된 윈도우 검증 완료 ✓")


if __name__ == '__main__':
    main()
