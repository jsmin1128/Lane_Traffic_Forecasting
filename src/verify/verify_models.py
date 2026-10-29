#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_models.py

목적: 기본 LSTM, ST-GCN, STGCNSegment 모델의 Forward PASS를
더미 입력으로 검증합니다.
"""

import torch
import numpy as np

# ─────────────────────────────────────────────────────────────────────────
# 공통 변수
# ─────────────────────────────────────────────────────────────────────────
B = 2           # 배치 크기
T = 12          # 시퀀스 길이
N = 1370        # 노드 수
C_traffic = 8   # Queue4+Speed4 채널 수
C_flag = 1      # 주말/공휴일 플래그 채널 수

# ─────────────────────────────────────────────────────────────────────────
# 1) 전통적 LSTM 모델
# ─────────────────────────────────────────────────────────────────────────
from traffic.models.lstm import BasicLSTM

C_lstm = C_traffic + C_flag
# (B, C_in, T, N)
x_lstm = torch.randn(B, C_lstm, T, N)

lstm_model = BasicLSTM(num_nodes=N, input_dim=C_traffic)
lstm_model.eval()
with torch.no_grad():
    y_lstm = lstm_model(x_lstm)
print("BasicLSTM Forward PASS 성공!  출력 shape:", y_lstm.shape)  # 예상: (B, 8, N)

# ─────────────────────────────────────────────────────────────────────────
# 2) 원본 ST-GCN 모델
# ─────────────────────────────────────────────────────────────────────────
from traffic.models.stgcn import STGCN

C_stgcn = C_traffic + C_flag
x_stgcn = torch.randn(B, C_stgcn, T, N)
A_dummy = torch.eye(N)

stgcn_model = STGCN(
        in_channels=C_stgcn,
        hidden1=64,
        out_channels=C_traffic,
        num_nodes=N,
        A=A_dummy
    )
stgcn_model.eval()
with torch.no_grad():
    y_stgcn = stgcn_model(x_stgcn)
print("ST-GCN Forward PASS 성공!  출력 shape:", y_stgcn.shape)  # 예상: (B, 8, N)

# ─────────────────────────────────────────────────────────────────────────
# 3) ST-GCN + Segment 모델
# ─────────────────────────────────────────────────────────────────────────
from traffic.models.stgcn_segment import STGCNSegment

C_seg = C_traffic + C_flag
x_seg = torch.randn(B, C_seg, T, N)
residual_seq = torch.randn(B, 1, C_traffic, N)   # (B, horizon, C, N)
cluster_id = torch.zeros(N, dtype=torch.long)

segment_model = STGCNSegment(
    in_c=C_traffic,   # forward()가 마지막 플래그 채널을 떼고 ST-GCN에 넘김
    out_c=C_traffic,
    num_nodes=N,
    A=A_dummy,
    cluster_id=cluster_id
)
segment_model.eval()
with torch.no_grad():
    y_segment = segment_model(x_seg, residual_seq=residual_seq)
print("STGCNSegment Forward PASS 성공!  출력 shape:", y_segment.shape)  # 예상: (B, 8, 1, N)
