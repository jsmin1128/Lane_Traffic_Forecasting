#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_training_curve.py

모델 하나의 학습·검증 손실 곡선(MSE, MAE)을 그립니다.

    outputs/<model>/metrics.csv → outputs/<model>/training_curve.png

Usage (저장소 루트에서 실행):
    python src/evaluate/plot_training_curve.py --model lstm
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import pandas as pd
import matplotlib.pyplot as plt

from model_io import LABELS, MODELS


def parse_args():
    p = argparse.ArgumentParser(description="Plot training and validation loss curves for one model")
    p.add_argument('--model', required=True, choices=MODELS)
    p.add_argument('--dpi',   type=int, default=150)
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = os.path.join('outputs', args.model)
    df = pd.read_csv(os.path.join(out_dir, 'metrics.csv'))   # epoch, train_mse, train_mae, val_mse, val_mae

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, loss in zip(axes, ['mse', 'mae']):
        ax.plot(df['epoch'], df[f'train_{loss}'], linestyle='--', marker='o', label='Train')
        ax.plot(df['epoch'], df[f'val_{loss}'],   linestyle='-',  marker='x', label='Val')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(f'{loss.upper()} Loss')
        ax.set_title(f'Training vs Validation {loss.upper()}')
        ax.legend()
        ax.grid(True)
    fig.suptitle(LABELS[args.model])
    fig.tight_layout()

    out_png = os.path.join(out_dir, 'training_curve.png')
    fig.savefig(out_png, dpi=args.dpi)
    print(f"Saved training curve to {out_png}")


if __name__ == '__main__':
    main()
