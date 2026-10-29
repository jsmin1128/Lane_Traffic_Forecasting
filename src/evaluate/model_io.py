#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
model_io.py

평가 스크립트들이 함께 쓰는 모델 목록, 체크포인트 찾기, 모델 불러오기, 예측 함수입니다.
"""

import glob
import os
import sys

import numpy as np
import torch

from traffic.models.lstm import BasicLSTM
from traffic.models.stgcn import STGCN

MODELS = ['lstm', 'stgcn', 'stgcn_segment']
LABELS = {'lstm': 'LSTM', 'stgcn': 'ST-GCN', 'stgcn_segment': 'ST-GCN + Segment'}

# 학습·평가 때 정답이 residual_seq로 들어가는 문제(정답 누설)가 고쳐질 때까지 평가하지 않는 모델
NOT_EVALUATED = {
    'stgcn_segment': "stgcn_segment는 정답이 입력으로 들어가는 문제(정답 누설)가 있어 평가 방식을 "
                     "고치기 전까지 평가하지 않습니다. outputs/stgcn_segment/NOTE.txt 참고",
}


def latest_checkpoint(model_name):
    """outputs/<model>/checkpoints/ 에서 가장 마지막 에폭의 체크포인트 경로"""
    ckpts = sorted(glob.glob(os.path.join('outputs', model_name, 'checkpoints', f'{model_name}_ep*.pt')))
    if not ckpts:
        sys.exit(f"체크포인트가 없습니다: outputs/{model_name}/checkpoints/")
    return ckpts[-1]


def load_model(model_name, ckpt_path, num_nodes, device):
    state = torch.load(ckpt_path, map_location=device)['model_state']
    if model_name == 'lstm':
        model = BasicLSTM(
            num_nodes=num_nodes,
            input_dim=8,
            hidden_dim=state['lstm.weight_ih_l0'].shape[0] // 4,
            num_layers=1,
            dropout=0.0
        )
    elif model_name == 'stgcn':
        A = torch.from_numpy(np.load('processed/graph/A_lane.npy')).float().to(device)
        model = STGCN(
            in_channels=9,
            hidden1=state['layer1.temp1.weight'].shape[0],
            out_channels=8,
            num_nodes=num_nodes,
            A=A,
            horizon=1
        )
    else:
        raise ValueError(f"평가를 지원하지 않는 모델입니다: {model_name}")
    model.load_state_dict(state)
    return model.to(device).eval()


def predict(model_name, model, x):
    """입력 (B, 9, 12, N) → 다음 5분 예측 (B, 8, N)"""
    out = model(x)
    return out if model_name == 'lstm' else out.squeeze(2)
