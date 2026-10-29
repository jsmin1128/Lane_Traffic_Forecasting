#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate.py

모델 하나를 평가합니다. 결과는 outputs/<model>/ 에 저장됩니다.
    training_curve.png, test_metrics.csv, figures/actual_vs_pred_<차종>.png

Usage (저장소 루트에서 실행):
    python scripts/evaluate.py lstm
    python scripts/evaluate.py stgcn --ckpt outputs/stgcn/checkpoints/stgcn_ep040.pt

--ckpt를 생략하면 가장 마지막 에폭의 체크포인트를 씁니다.
stgcn_segment는 정답 누설 문제가 있어 학습 곡선만 그립니다 (README 참고).
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from traffic import evaluation
from traffic.models import MODEL_NAMES


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate one model")
    p.add_argument('model', choices=MODEL_NAMES)
    p.add_argument('--ckpt',    type=str, default=None, help='Checkpoint path (default: last epoch)')
    p.add_argument('--batch',   type=int, default=8,    help='Test batch size')
    p.add_argument('--workers', type=int, default=None, help='DataLoader 작업자 수 (기본: 자동, 0이면 메인 프로세스)')
    p.add_argument('--dpi',     type=int, default=150)
    return p.parse_args()


def main():
    args = parse_args()
    evaluation.plot_training_curve(args.model, args.dpi)
    if args.model in evaluation.NOT_EVALUATED:
        print(evaluation.NOT_EVALUATED[args.model])
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt_path = args.ckpt or evaluation.latest_checkpoint(args.model)
    y_true, y_pred, window_idx, dates = evaluation.predict_test_set(args.model, ckpt_path, args.batch,
                                                                    args.workers, device)
    evaluation.save_test_metrics(args.model, ckpt_path, y_true, y_pred)
    evaluation.plot_actual_vs_pred(args.model, y_true, y_pred, window_idx, dates, args.dpi)


if __name__ == '__main__':
    main()
