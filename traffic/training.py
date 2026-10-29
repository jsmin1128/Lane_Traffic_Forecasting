"""
세 모델 공통 학습 루프입니다. scripts/train.py가 모델 이름과 설정을 넘겨 호출합니다.

- 손실: MSE로 학습하고, 매 에폭 학습·검증 MSE·MAE를 outputs/<model>/metrics.csv에 기록
- 체크포인트: save_freq 에폭마다 또는 검증 MSE가 좋아질 때 outputs/<model>/checkpoints/<model>_epNNN.pt
- 조기 종료: 검증 MSE가 patience 에폭 동안 좋아지지 않으면 멈춤 (patience=None이면 끔)
"""

import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from traffic import config
from traffic.data import get_dataloaders
from traffic.models import BasicLSTM, STGCN, STGCNSegment

# 모델별 기본 설정 (원래 모델별 학습 스크립트의 기본값 그대로)
DEFAULTS = {
    'lstm': dict(epochs=40, batch=8, lr=5e-4, clip=5.0, save_freq=5, patience=10,
                 sched='plateau', plateau_patience=3,
                 hidden_dim=64, num_layers=1, dropout=0.0),
    'stgcn': dict(epochs=40, batch=8, lr=5e-4, clip=5.0, save_freq=5, patience=10,
                  sched='plateau', plateau_patience=3,
                  hidden=64),
    # 원래 스크립트의 조기 종료는 버그로 작동하지 않았으므로 기본값은 끔(patience=None)
    'stgcn_segment': dict(epochs=30, batch=8, lr=5e-4, clip=5.0, save_freq=5, patience=None,
                          sched='plateau', plateau_patience=5,
                          hidden_lstm=256, horizon=1, cluster_map=None),
}


def build_model(name, opts, num_nodes, device):
    if name == 'lstm':
        return BasicLSTM(num_nodes=num_nodes,
                         input_dim=config.NUM_TRAFFIC,
                         hidden_dim=opts['hidden_dim'],
                         num_layers=opts['num_layers'],
                         dropout=opts['dropout'])

    A = torch.from_numpy(np.load(config.PROCESSED_DIR / 'graph' / 'A_lane.npy')).float().to(device)
    if name == 'stgcn':
        return STGCN(in_channels=config.NUM_TRAFFIC + 1,
                     hidden1=opts['hidden'],
                     out_channels=config.NUM_TRAFFIC,
                     num_nodes=A.size(0),
                     A=A,
                     horizon=1)

    cluster_map = opts['cluster_map'] or config.PROCESSED_DIR / 'graph' / 'lane_to_segment_id_by_edge.npy'
    cluster_id = torch.from_numpy(np.load(cluster_map)).long().to(device)
    return STGCNSegment(in_c=config.NUM_TRAFFIC,
                        out_c=config.NUM_TRAFFIC,
                        num_nodes=num_nodes,
                        A=A,
                        cluster_id=cluster_id,
                        horizon=opts['horizon'],
                        K=int(cluster_id.max()) + 1,
                        hidden_lstm=opts['hidden_lstm'])


def forward(name, model, batch, device, opts):
    """배치 하나 → (예측, 정답)"""
    x, y_last, y_seq, *_ = batch
    x = x.to(device)
    if name == 'lstm':
        return model(x), y_last.to(device)
    if name == 'stgcn':
        return model(x).squeeze(2), y_last.to(device)
    # stgcn_segment: 원래 학습 방식 그대로 — 입력 구간 첫 스텝을 정답으로 쓰고 residual_seq로도 넣음 (정답 누설, README 참고)
    y_true = y_seq[:, :opts['horizon']].to(device)
    return model(x, residual_seq=y_true).permute(0, 2, 1, 3), y_true


def save_ckpt(path, epoch, model, val_mse, val_mae):
    torch.save({'epoch': epoch,
                'model_state': model.state_dict(),
                'val_mse': val_mse,
                'val_mae': val_mae}, path)


def train(name, opts, num_workers=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"▶ Model: {name} | device: {device}")

    # Data
    train_loader, val_loader, _ = get_dataloaders(batch_size=opts['batch'], num_workers=num_workers)
    num_nodes = train_loader.dataset.X.shape[2]

    # Model
    model = build_model(name, opts, num_nodes, device).to(device)
    print(f"▶ Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Losses & optimiser
    mse_criterion = nn.MSELoss()
    mae_criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=opts['lr'], weight_decay=1e-4)
    scheduler = (optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5,
                                                      patience=opts['plateau_patience'])
                 if opts['sched'] == 'plateau'
                 else optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opts['epochs'], eta_min=1e-5))

    # Output dirs
    out_dir = os.path.join(config.OUTPUTS_DIR, name)
    ckpt_dir = os.path.join(out_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, 'metrics.csv')
    with open(csv_path, 'w') as f:
        f.write('epoch,train_mse,train_mae,val_mse,val_mae\n')

    best_val = float('inf')
    patience_counter = 0

    # ──────────────── Training loop ────────────────
    for ep in range(1, opts['epochs'] + 1):
        # -- Train ---------------------------------------------------------
        model.train()
        train_mse_sum, train_mae_sum = 0.0, 0.0
        for batch in tqdm(train_loader, desc=f"[Train] Ep{ep}", ncols=80):
            pred, target = forward(name, model, batch, device, opts)
            loss_mse = mse_criterion(pred, target)
            loss_mae = mae_criterion(pred, target)

            optimizer.zero_grad()
            loss_mse.backward()   # MSE로 학습
            if opts['clip'] > 0:
                nn.utils.clip_grad_norm_(model.parameters(), opts['clip'])
            optimizer.step()

            train_mse_sum += loss_mse.item()
            train_mae_sum += loss_mae.item()
        train_mse = train_mse_sum / len(train_loader)
        train_mae = train_mae_sum / len(train_loader)

        # -- Validation ----------------------------------------------------
        model.eval()
        val_mse_sum, val_mae_sum = 0.0, 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"[Val]   Ep{ep}", ncols=80):
                pred, target = forward(name, model, batch, device, opts)
                val_mse_sum += mse_criterion(pred, target).item()
                val_mae_sum += mae_criterion(pred, target).item()
        val_mse = val_mse_sum / len(val_loader)
        val_mae = val_mae_sum / len(val_loader)

        # -- Scheduler -----------------------------------------------------
        if opts['sched'] == 'plateau':
            scheduler.step(val_mse)
        else:
            scheduler.step()

        # -- Logging -------------------------------------------------------
        print(f"[Ep{ep:02d}] Train MSE {train_mse:.6f} | Val MSE {val_mse:.6f} || Train MAE {train_mae:.6f} | Val MAE {val_mae:.6f}")
        with open(csv_path, 'a') as f:
            f.write(f"{ep},{train_mse:.6f},{train_mae:.6f},{val_mse:.6f},{val_mae:.6f}\n")

        # -- Checkpoint ----------------------------------------------------
        if ep % opts['save_freq'] == 0 or val_mse < best_val:
            ckpt_path = os.path.join(ckpt_dir, f"{name}_ep{ep:03d}.pt")
            save_ckpt(ckpt_path, ep, model, val_mse, val_mae)
            print(f"▶ Saved checkpoint: {ckpt_path}")

        # -- Early Stop ----------------------------------------------------
        if val_mse < best_val:
            best_val = val_mse
            patience_counter = 0
        else:
            patience_counter += 1
            if opts['patience'] is not None and patience_counter >= opts['patience']:
                print(f"▶ Early stopping at epoch {ep}")
                break

    print(f"▶ Training complete. Outputs under {out_dir}/")
