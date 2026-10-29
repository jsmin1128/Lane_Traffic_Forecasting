"""모델 3개와, 학습·평가 스크립트에서 모델을 고를 때 쓰는 이름."""

from traffic.models.lstm import BasicLSTM
from traffic.models.stgcn import STGCN
from traffic.models.stgcn_segment import STGCNSegment

MODEL_NAMES = ['lstm', 'stgcn', 'stgcn_segment']
LABELS = {'lstm': 'LSTM', 'stgcn': 'ST-GCN', 'stgcn_segment': 'ST-GCN + Segment'}
