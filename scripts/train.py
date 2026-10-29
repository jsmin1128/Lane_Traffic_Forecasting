#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train.py

모델 하나를 학습합니다. 결과는 outputs/<model>/ 에 저장됩니다.
    checkpoints/<model>_epNNN.pt, metrics.csv

Usage (저장소 루트에서 실행):
    python scripts/train.py lstm
    python scripts/train.py stgcn --epochs 80
    python scripts/train.py stgcn_segment --hidden_lstm 64

모델별 기본값은 traffic/training.py의 DEFAULTS에 있고, 아래 옵션으로 덮어쓸 수 있습니다.
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from traffic.models import MODEL_NAMES
from traffic.training import DEFAULTS, train


def parse_args():
    p = argparse.ArgumentParser(description="Train one model")
    p.add_argument('model', choices=MODEL_NAMES)
    p.add_argument('--epochs',    type=int)
    p.add_argument('--batch',     type=int)
    p.add_argument('--lr',        type=float)
    p.add_argument('--clip',      type=float, help='Gradient clipping norm (0이면 끔)')
    p.add_argument('--save_freq', type=int,   help='N 에폭마다 체크포인트 저장')
    p.add_argument('--patience',  type=int,   help='조기 종료 patience')
    p.add_argument('--sched',     choices=['plateau', 'cosine'], help='LR scheduler')
    p.add_argument('--workers',   type=int,   help='DataLoader 작업자 수 (기본: 자동, 0이면 메인 프로세스)')
    # 모델별 옵션
    p.add_argument('--hidden_dim',  type=int,   help='[lstm] LSTM hidden 크기')
    p.add_argument('--num_layers',  type=int,   help='[lstm] LSTM 층 수')
    p.add_argument('--dropout',     type=float, help='[lstm] LSTM 층 사이 dropout')
    p.add_argument('--hidden',      type=int,   help='[stgcn] hidden 채널 수')
    p.add_argument('--hidden_lstm', type=int,   help='[stgcn_segment] Cluster-LSTM hidden 크기')
    p.add_argument('--horizon',     type=int,   help='[stgcn_segment] 예측 스텝 수')
    p.add_argument('--cluster_map', type=str,   help='[stgcn_segment] 세그먼트 매핑 .npy 경로')
    args = p.parse_args()

    opts = dict(DEFAULTS[args.model])
    for key, value in vars(args).items():
        if key in ('model', 'workers') or value is None:
            continue
        if key not in opts:
            p.error(f"{args.model} 모델에는 --{key} 옵션이 없습니다")
        opts[key] = value
    return args, opts


def main():
    args, opts = parse_args()
    train(args.model, opts, num_workers=args.workers)


if __name__ == '__main__':
    main()
