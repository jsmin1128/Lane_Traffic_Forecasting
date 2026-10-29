#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_actual_vs_pred.py

모델 하나의 테스트셋 예측을 실제 단위로 되돌려, 하루 시간대(5분 슬롯)별로
실제 대기 차량 수와 비교한 그래프를 차종별로 저장합니다.
값은 1370개 차로의 합계를 날짜별로 평균한 것입니다.

    → outputs/<model>/figures/actual_vs_pred_{car,bus,truck,motorcycle}.png

Usage (저장소 루트에서 실행):
    python src/evaluate/plot_actual_vs_pred.py --model stgcn

--ckpt를 생략하면 가장 마지막 에폭의 체크포인트를 씁니다.
"""

import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
import matplotlib.pyplot as plt

from traffic.data import get_dataloaders
from model_io import LABELS, MODELS, NOT_EVALUATED, latest_checkpoint, load_model, predict

NUM_SLOTS       = 288                       # 하루 5분 슬롯 수
WINDOW_SIZE     = 12                        # 입력 12스텝(1시간)
WINDOWS_PER_DAY = NUM_SLOTS - WINDOW_SIZE   # 하루 윈도우 수 (make_windows.py와 동일)
CATEGORIES      = ['Car', 'Bus', 'Truck', 'Motorcycle']   # 채널 0~3: 차종별 대기 차량 수


def parse_args():
    p = argparse.ArgumentParser(description="Plot actual vs predicted queue counts by time of day")
    p.add_argument('--model', required=True, choices=MODELS)
    p.add_argument('--ckpt',  type=str, default=None, help='Checkpoint path (default: last epoch)')
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--dpi',   type=int, default=150)
    return p.parse_args()


def date_stats(date):
    """build_tensor.py가 정규화에 쓴 날짜별 채널 평균·표준편차"""
    raw = np.load(os.path.join('processed', 'tensor', str(date), f'input_tensor_{date}.npy'))  # (288,1370,8)
    ch = raw.reshape(-1, raw.shape[-1])
    return ch.mean(axis=0), ch.std(axis=0)


def main():
    args = parse_args()
    if args.model in NOT_EVALUATED:
        sys.exit(NOT_EVALUATED[args.model])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt_path = args.ckpt or latest_checkpoint(args.model)

    train_loader, val_loader, test_loader = get_dataloaders(batch_size=args.batch)
    offset = len(train_loader.dataset) + len(val_loader.dataset)   # 테스트셋 첫 윈도우의 전체 인덱스
    model = load_model(args.model, ckpt_path, test_loader.dataset.X.shape[2], device)
    print(f"Predicting with {args.model} ({ckpt_path}) …")

    stats = {}
    slot_true, slot_pred = defaultdict(list), defaultdict(list)
    with torch.no_grad():
        for x, y_last, _, idx, date in test_loader:
            pred = predict(args.model, model, x.to(device)).cpu().numpy()   # (B,8,N)
            true = y_last.numpy()                                           # (B,8,N)
            for i in range(len(idx)):
                d = int(date[i])
                if d not in stats:
                    stats[d] = date_stats(d)
                means, stds = stats[d]
                slot = (offset + int(idx[i])) % WINDOWS_PER_DAY + WINDOW_SIZE   # 예측 대상 슬롯
                # 정규화를 되돌린 뒤 차로 합계 (채널 0~3)
                slot_true[slot].append((true[i, :4] * stds[:4, None] + means[:4, None]).sum(axis=1))
                slot_pred[slot].append((pred[i, :4] * stds[:4, None] + means[:4, None]).sum(axis=1))

    slots = np.array(sorted(slot_true))
    true_mean = np.array([np.mean(slot_true[s], axis=0) for s in slots])   # (slots,4)
    pred_mean = np.array([np.mean(slot_pred[s], axis=0) for s in slots])
    labels = [f"{s * 5 // 60:02d}:{s * 5 % 60:02d}" for s in slots]

    out_dir = os.path.join('outputs', args.model, 'figures')
    os.makedirs(out_dir, exist_ok=True)
    for i, name in enumerate(CATEGORIES):
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(slots, true_mean[:, i], label=f"Actual {name}")
        ax.plot(slots, pred_mean[:, i], '--', label=f"Predicted {name}")
        ax.set_xticks(slots[::12])
        ax.set_xticklabels(labels[::12], rotation=45)
        ax.set_xlabel('Time (HH:MM)')
        ax.set_ylabel('Average Queue Count (sum of lanes)')
        ax.set_title(f'{LABELS[args.model]}: Actual vs Predicted Queue Count for {name} (test set)')
        ax.legend()
        ax.grid(True)
        fig.tight_layout()

        out_file = os.path.join(out_dir, f'actual_vs_pred_{name.lower()}.png')
        fig.savefig(out_file, dpi=args.dpi)
        print(f"Saved {name} plot to {out_file}")
        plt.close(fig)


if __name__ == '__main__':
    main()
