#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
check_data_loader.py

목적: traffic/data.py의 get_dataloaders()가 제대로 작동하는지 검증
  - train/val/test 윈도우 수와 배치 shape
  - 학습 데이터 채널별 평균·표준편차 (정규화 확인)

사용법: python scripts/check/check_data_loader.py [--workers 0]
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from traffic.data import get_dataloaders


def main():
    p = argparse.ArgumentParser(description="Check get_dataloaders()")
    p.add_argument('--batch',   type=int, default=4)
    p.add_argument('--workers', type=int, default=None, help='DataLoader 작업자 수 (기본: 자동)')
    args = p.parse_args()

    print(f"▶ get_dataloaders(batch_size={args.batch}) 호출 중...")
    loaders = dict(zip(['train', 'val', 'test'],
                       get_dataloaders(batch_size=args.batch, num_workers=args.workers)))

    # --- 분할별 첫 배치 ---
    for name, loader in loaders.items():
        x, y_last, y_seq, idx, date = next(iter(loader))
        print(f"{name:5s}: {len(loader.dataset):5d} windows | x {tuple(x.shape)} | y_last {tuple(y_last.shape)} "
              f"| y_seq {tuple(y_seq.shape)} | first date {int(date[0])}")

    # --- 학습 데이터 채널별 평균/표준편차 (정규화 확인) ---
    total = sq_total = 0.0
    count = 0
    for x, *_ in loaders['train']:
        x = x.double().numpy()                       # (B, C, T, N)
        total = total + x.sum(axis=(0, 2, 3))
        sq_total = sq_total + (x ** 2).sum(axis=(0, 2, 3))
        count += x.shape[0] * x.shape[2] * x.shape[3]
    mean = total / count
    std = np.sqrt(sq_total / count - mean ** 2)
    for ch in range(len(mean)):
        print(f"  채널 {ch:2d}: 평균 = {mean[ch]:.4f}, 표준편차 = {std[ch]:.4f}")

    print("\n▶ 데이터 로더 검증 완료!")


if __name__ == '__main__':
    main()
