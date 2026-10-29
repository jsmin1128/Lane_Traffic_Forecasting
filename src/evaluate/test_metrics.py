#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_metrics.py

모델 하나를 테스트셋(마지막 15% 윈도우)으로 평가해 MAE, RMSE, R²를 저장합니다.
지표는 정규화(z-score) 값 기준입니다.

    outputs/<model>/checkpoints/ → outputs/<model>/test_metrics.csv

Usage (저장소 루트에서 실행):
    python src/evaluate/test_metrics.py --model lstm
    python src/evaluate/test_metrics.py --model stgcn --ckpt outputs/stgcn/checkpoints/stgcn_ep040.pt

--ckpt를 생략하면 가장 마지막 에폭의 체크포인트를 씁니다.
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from traffic.data import get_dataloaders
from model_io import MODELS, NOT_EVALUATED, latest_checkpoint, load_model, predict


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate one model on the test set")
    p.add_argument('--model', required=True, choices=MODELS)
    p.add_argument('--ckpt',  type=str, default=None, help='Checkpoint path (default: last epoch)')
    p.add_argument('--batch', type=int, default=8,    help='Test batch size')
    return p.parse_args()


def main():
    args = parse_args()
    if args.model in NOT_EVALUATED:
        sys.exit(NOT_EVALUATED[args.model])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt_path = args.ckpt or latest_checkpoint(args.model)

    _, _, test_loader = get_dataloaders(batch_size=args.batch)
    model = load_model(args.model, ckpt_path, test_loader.dataset.X.shape[2], device)
    print(f"Evaluating {args.model} ({ckpt_path}) …")

    y_true_all, y_pred_all = [], []
    with torch.no_grad():
        for x, y_last, *_ in test_loader:
            pred = predict(args.model, model, x.to(device)).cpu().numpy()   # (B,8,N)
            y_pred_all.append(pred.reshape(-1))
            y_true_all.append(y_last.numpy().reshape(-1))
    y_pred = np.concatenate(y_pred_all)
    y_true = np.concatenate(y_true_all)

    out_csv = os.path.join('outputs', args.model, 'test_metrics.csv')
    pd.DataFrame([{
        'model':      args.model,
        'checkpoint': os.path.basename(ckpt_path),
        'mae':        mean_absolute_error(y_true, y_pred),
        'rmse':       np.sqrt(mean_squared_error(y_true, y_pred)),
        'r2':         r2_score(y_true, y_pred),
    }]).to_csv(out_csv, index=False)
    print(f"Saved test metrics to {out_csv}")


if __name__ == '__main__':
    main()
