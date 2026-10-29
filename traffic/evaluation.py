"""
모델 하나의 평가 결과를 만듭니다 (outputs/<model>/).

  training_curve.png                 학습·검증 손실 곡선 (metrics.csv에서)
  test_metrics.csv                   테스트셋(마지막 15% 윈도우) MAE, RMSE, R² — 정규화(z-score) 값 기준
  figures/actual_vs_pred_<차종>.png   하루 시간대별 실제 vs 예측 대기 차량 수 (테스트셋, 실제 단위)
"""

import glob
import os
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from traffic import config
from traffic.data import get_dataloaders
from traffic.models import LABELS, BasicLSTM, STGCN

# 학습·평가 때 정답이 residual_seq로 들어가는 문제(정답 누설)가 고쳐질 때까지 테스트 평가를 하지 않는 모델
NOT_EVALUATED = {
    'stgcn_segment': "stgcn_segment는 정답이 입력으로 들어가는 문제(정답 누설)가 있어 평가 방식을 "
                     "고치기 전까지 테스트 지표와 그래프를 만들지 않습니다. README 참고",
}

WINDOWS_PER_DAY = config.NUM_SLOTS - config.WINDOW_SIZE   # 하루 윈도우 수
CATEGORIES      = ['Car', 'Bus', 'Truck', 'Motorcycle']   # 채널 0~3: 차종별 대기 차량 수


def latest_checkpoint(name):
    """outputs/<model>/checkpoints/ 에서 가장 마지막 에폭의 체크포인트 경로"""
    ckpts = sorted(glob.glob(os.path.join(config.OUTPUTS_DIR, name, 'checkpoints', f'{name}_ep*.pt')))
    if not ckpts:
        sys.exit(f"체크포인트가 없습니다: outputs/{name}/checkpoints/")
    return ckpts[-1]


def load_model(name, ckpt_path, num_nodes, device):
    state = torch.load(ckpt_path, map_location=device)['model_state']
    if name == 'lstm':
        model = BasicLSTM(
            num_nodes=num_nodes,
            input_dim=config.NUM_CHANNELS,
            hidden_dim=state['lstm.weight_ih_l0'].shape[0] // 4,
            num_layers=1,
            dropout=0.0
        )
    elif name == 'stgcn':
        A = torch.from_numpy(np.load(config.PROCESSED_DIR / 'graph' / 'adjacency.npy')).float().to(device)
        model = STGCN(
            in_channels=config.NUM_CHANNELS + 1,
            hidden1=state['layer1.temp1.weight'].shape[0],
            out_channels=config.NUM_CHANNELS,
            num_nodes=num_nodes,
            A=A,
            horizon=1
        )
    else:
        raise ValueError(f"평가를 지원하지 않는 모델입니다: {name}")
    model.load_state_dict(state)
    return model.to(device).eval()


def predict(name, model, x):
    """입력 (B, 9, 12, N) → 다음 5분 예측 (B, 8, N)"""
    out = model(x)
    return out if name == 'lstm' else out.squeeze(2)


def predict_test_set(name, ckpt_path, batch, num_workers, device):
    """테스트셋 전체를 예측합니다 → (정답, 예측, 윈도우 전체 인덱스, 날짜). 정답·예측 shape: (윈도우 수, 8, N)"""
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=batch, num_workers=num_workers)
    offset = len(train_loader.dataset) + len(val_loader.dataset)   # 테스트셋 첫 윈도우의 전체 인덱스
    model = load_model(name, ckpt_path, test_loader.dataset.X.shape[2], device)
    print(f"Predicting with {name} ({ckpt_path}) …")

    trues, preds, idxs, dates = [], [], [], []
    with torch.no_grad():
        for x, y_last, _, idx, date in test_loader:
            preds.append(predict(name, model, x.to(device)).cpu().numpy())
            trues.append(y_last.numpy())
            idxs.append(idx.numpy())
            dates.append(date.numpy())
    return np.concatenate(trues), np.concatenate(preds), offset + np.concatenate(idxs), np.concatenate(dates)


def save_test_metrics(name, ckpt_path, y_true, y_pred):
    y_true, y_pred = y_true.reshape(-1), y_pred.reshape(-1)
    out_csv = os.path.join(config.OUTPUTS_DIR, name, 'test_metrics.csv')
    pd.DataFrame([{
        'model':      name,
        'checkpoint': os.path.basename(ckpt_path),
        'mae':        mean_absolute_error(y_true, y_pred),
        'rmse':       np.sqrt(mean_squared_error(y_true, y_pred)),
        'r2':         r2_score(y_true, y_pred),
    }]).to_csv(out_csv, index=False)
    print(f"Saved test metrics to {out_csv}")


def date_stats(date):
    """정규화를 되돌릴 때 쓰는 날짜별 채널 평균·표준편차 (tensors/values.npy에서 계산)"""
    tensors_dir = config.PROCESSED_DIR / 'tensors'
    dates = np.load(tensors_dir / 'dates.npy').tolist()
    values = np.load(tensors_dir / 'values.npy', mmap_mode='r')[dates.index(date)]   # (288, N, 8)
    ch = np.asarray(values).reshape(-1, values.shape[-1])
    return ch.mean(axis=0), ch.std(axis=0)


def plot_actual_vs_pred(name, y_true, y_pred, window_idx, dates, dpi=150):
    """정규화를 되돌린 대기 차량 수(차로 합계)를 하루 시간대(5분 슬롯)별로 평균해 차종별로 그립니다."""
    stats = {}
    slot_true, slot_pred = defaultdict(list), defaultdict(list)
    for true, pred, gidx, d in zip(y_true, y_pred, window_idx, dates):
        d = int(d)
        if d not in stats:
            stats[d] = date_stats(d)
        means, stds = stats[d]
        slot = int(gidx) % WINDOWS_PER_DAY + config.WINDOW_SIZE   # 예측 대상 슬롯
        # 정규화를 되돌린 뒤 차로 합계 (채널 0~3)
        slot_true[slot].append((true[:4] * stds[:4, None] + means[:4, None]).sum(axis=1))
        slot_pred[slot].append((pred[:4] * stds[:4, None] + means[:4, None]).sum(axis=1))

    slots = np.array(sorted(slot_true))
    true_mean = np.array([np.mean(slot_true[s], axis=0) for s in slots])   # (slots,4)
    pred_mean = np.array([np.mean(slot_pred[s], axis=0) for s in slots])
    labels = [f"{s * 5 // 60:02d}:{s * 5 % 60:02d}" for s in slots]

    out_dir = os.path.join(config.OUTPUTS_DIR, name, 'figures')
    os.makedirs(out_dir, exist_ok=True)
    for i, cat in enumerate(CATEGORIES):
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(slots, true_mean[:, i], label=f"Actual {cat}")
        ax.plot(slots, pred_mean[:, i], '--', label=f"Predicted {cat}")
        ax.set_xticks(slots[::12])
        ax.set_xticklabels(labels[::12], rotation=45)
        ax.set_xlabel('Time (HH:MM)')
        ax.set_ylabel('Average Queue Count (sum of lanes)')
        ax.set_title(f'{LABELS[name]}: Actual vs Predicted Queue Count for {cat} (test set)')
        ax.legend()
        ax.grid(True)
        fig.tight_layout()

        out_file = os.path.join(out_dir, f'actual_vs_pred_{cat.lower()}.png')
        fig.savefig(out_file, dpi=dpi)
        print(f"Saved {cat} plot to {out_file}")
        plt.close(fig)


def plot_training_curve(name, dpi=150):
    """outputs/<model>/metrics.csv → outputs/<model>/training_curve.png"""
    out_dir = os.path.join(config.OUTPUTS_DIR, name)
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
    fig.suptitle(LABELS[name])
    fig.tight_layout()

    out_png = os.path.join(out_dir, 'training_curve.png')
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)
    print(f"Saved training curve to {out_png}")
