#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
check_tensors.py

processed/tensors의 values.npy, normalized.npy, dates.npy를 점검합니다 (파일은 만들지 않음).
  - shape, dtype, NaN·Inf·음수 개수
  - 채널별 실제 값 통계 (31일 전체): 최소, 최대, 평균, 0의 비율
  - normalized가 날짜·채널마다 평균 0, 표준편차 1인지 (값이 모두 같은 채널은 전부 0)

사용법: python scripts/check/check_tensors.py
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd

from traffic import config

CHANNELS = ['대기 승용', '대기 버스', '대기 트럭', '대기 오토바이', '속도 승용', '속도 버스', '속도 트럭', '속도 오토바이']


def main():
    tensors_dir = config.PROCESSED_DIR / 'tensors'
    values = np.load(tensors_dir / 'values.npy')
    normalized = np.load(tensors_dir / 'normalized.npy')
    dates = np.load(tensors_dir / 'dates.npy')
    print(f"▶ values {values.shape} {values.dtype} | normalized {normalized.shape} {normalized.dtype} | "
          f"dates {dates.shape} ({dates[0]} ~ {dates[-1]})")
    assert values.shape == normalized.shape and values.shape[0] == len(dates), "배열 크기가 서로 맞지 않습니다."
    for name, x in [('values', values), ('normalized', normalized)]:
        print(f"▶ {name}: NaN {int(np.isnan(x).sum())}개, Inf {int(np.isinf(x).sum())}개, 음수 {int((x < 0).sum())}개")

    flat = values.reshape(-1, values.shape[-1])
    stats = pd.DataFrame({'최소': flat.min(0), '최대': flat.max(0), '평균': flat.mean(0, dtype=np.float64),
                          '0의 비율': (flat == 0).mean(0)}, index=CHANNELS)
    print("\n▶ 채널별 실제 값 (31일 전체)")
    print(stats.round(3).to_string())

    # 날짜·채널마다 평균 0, 표준편차 1 (표준편차 0인 채널은 전부 0)
    mean = normalized.mean(axis=(1, 2), dtype=np.float64)
    std = normalized.std(axis=(1, 2), dtype=np.float64)
    ok = (np.abs(mean) < 1e-3) & ((np.abs(std - 1) < 1e-3) | (std == 0))
    print(f"\n▶ 정규화 확인: 날짜×채널 {ok.size}개 중 {int(ok.sum())}개 통과")
    for d, c in zip(*np.nonzero(~ok)):
        print(f"  ✗ {dates[d]} {CHANNELS[c]}: 평균 {mean[d, c]:.4f}, 표준편차 {std[d, c]:.4f}")


if __name__ == '__main__':
    main()
