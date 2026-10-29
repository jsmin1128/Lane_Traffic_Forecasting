# 부천시 차로 단위 교통 예측

부천시 교차로 주변 1,370개 차로의 교통 상태를 5분 단위로 예측하는 프로젝트입니다. 과거 1시간(5분 × 12스텝)의
차종별 대기 차량 수·평균 속도와 주말·공휴일 여부를 보고 다음 5분을 예측하며, LSTM, ST-GCN, ST-GCN + Segment
세 모델을 하나씩 학습·비교합니다.

데이터는 AI Hub의 「AI 기반 신호 최적화를 위한 데이터」(2-058) 중 2022년 8~10월의 31일치를 씁니다.

## 요구 사항

| 항목 | 버전 |
|---|---|
| Python | 3.13 |
| PyTorch | 2.13.0+cu126 |
| GPU | 선택 (CUDA를 쓸 수 있으면 GPU, 없으면 CPU로 자동 선택) |

## 설치

```bash
pip install -r requirements.txt
```

`traffic/`이 저장소 루트에 있어 스크립트가 별도 패키지 설치 없이 바로 import합니다.

> **torch는 CUDA 12.x 드라이버용 `+cu126` 휠로 고정돼 있습니다.** 드라이버가 다르면 `nvidia-smi`로 확인해
> `requirements.txt`의 index-url과 태그를 맞게 바꾸세요.

## 빠른 시작

```bash
# 1. data/raw/ 준비 (아래 "데이터" 참고)

# 2. 전처리: data/raw → processed/ (약 5분)
python scripts/build_dataset.py

# 3. 학습 → 평가 (모델 하나씩, 결과는 outputs/<모델명>/)
python scripts/train.py stgcn
python scripts/evaluate.py stgcn
```

모든 스크립트는 저장소 루트에서 실행합니다.

## 프로젝트 구조

```text
.
├── data/raw/                      # 원본 데이터 (git 제외)
├── processed/                     # 전처리 결과 (git 제외, build_dataset.py로 다시 만듦)
├── outputs/{모델명}/              # 모델별 학습·평가 결과 (git 제외)
│
├── scripts/                       # 진입점 (전부 저장소 루트에서 실행)
│   ├── build_dataset.py           # 전처리 data/raw → processed/
│   ├── train.py                   # 학습 (train.py <모델명>)
│   ├── evaluate.py                # 평가 (evaluate.py <모델명>)
│   ├── build_sumo.py              # (선택) SUMO 시뮬레이션 입력 만들기
│   ├── etl/                       # AI Hub 원본 → data/raw 변환 (기록용)
│   └── check/                     # 점검 도구 (parquet, 그래프, 세그먼트, 텐서, 윈도우, 데이터 로더, 모델)
│
├── traffic/                       # 라이브러리 (설치 불필요, 루트에서 import)
│   ├── config.py                  # ★ 경로, 날짜 31일, 상수
│   ├── data.py                    # Dataset, get_dataloaders (70/15/15 분할)
│   ├── models/                    # lstm.py, stgcn.py, stgcn_segment.py
│   ├── preprocess/                # graph, segments, tensors, windows, sumo
│   ├── training.py                # 공통 학습 루프, 모델별 기본 설정 (DEFAULTS)
│   └── evaluation.py              # 테스트 지표, 그래프
│
├── notebooks/build_tensor.ipynb   # 텐서 만드는 과정을 하루치(2022-08-10)로 따라가는 노트북
├── requirements.txt
└── README.md
```

## 데이터

`data/raw/`에 AI Hub 원본을 Parquet으로 변환한 데이터를 둡니다 (약 3.5GB). Parquet은 모두
route(교차로·구간) 13개 × 날짜 31일로 나뉘어 있습니다.

```text
data/raw/
├── od.parquet/            # 기종점 교통량: 차량별 시각·위치·속도·차종·차로·대기 여부   → 텐서
├── turn.parquet/          # 회전 교통량: 구간 간 차종별 통과 대수                      → SUMO
├── signal_xml.parquet/    # 신호 현시 계획                                             → SUMO
├── signal_csv.parquet/    # 신호 상태 기록
├── indicator.parquet/     # 구간별 평균 속도·지체 시간·통행 시간
└── net/                   # SUMO 도로망 bucheon.net.xml, 차종 정의 exp.rou.xml 등      → 그래프, SUMO
```

AI Hub 데이터는 재배포할 수 없어 git에 올리지 않습니다. 따로 보관하세요. 변환에 쓴 코드는 `scripts/etl/`에
있고, AI Hub 원본 폴더를 인자로 받습니다 (예: `python scripts/etl/etl_od.py <교통량데이터(기종점교통량기반) 폴더>`).
`net/`은 AI Hub 네트워크데이터를 그대로 복사한 것이며, 도로망 파일 이름만 `1200012n.net.xml` → `bucheon.net.xml`로
바꿨습니다 (`bucheon.sumocfg`의 참조도 함께).

## 전처리

```bash
python scripts/build_dataset.py                  # 전부
python scripts/build_dataset.py flags windows    # 일부 단계만 (뒤 단계는 앞 단계 결과를 읽음)
```

| 단계 | 결과 (`processed/`) | 내용 |
|---|---|---|
| graph | `graph/lanes.txt`, `A_lane.npy` | 차로 목록, 정규화된 차로 인접행렬 (1370×1370) |
| segments | `graph/lane_to_segment_id_by_edge.npy` | 차로 → 세그먼트(88개) 매핑 |
| tensors | `tensor/<날짜>/input_tensor_<날짜>.npy`, `normalized_tensor_<날짜>.npy` | 날짜별 (288 슬롯, 1370 차로, 8채널), 정규화는 날짜·채널별 z-score |
| flags | `tensor/<날짜>/*_with_flag.npy` | 주말·공휴일 플래그 채널 추가 (9채널) |
| windows | `windows/all_X.npy`, `all_Y.npy`, `all_DATE.npy` | X (8556, 12, 1370, 9), Y (8556, 1370, 8) |

8채널은 차종 4종(승용, 버스, 트럭, 오토바이)별 대기 차량 수와 평균 속도입니다. 윈도우는 시간 순서대로
학습·검증·테스트 70/15/15 (5,989 / 1,283 / 1,284개)로 나누며, 테스트셋은 2022-10-26 09:00 ~ 10-31 23:55(예측 시점 기준)입니다.

## 학습

| 모델 | 이름 | 구조 | 기본 설정 |
|---|---|---|---|
| LSTM | `lstm` | 차로마다 같은 LSTM으로 과거 1시간 8채널을 보고 예측. 도로 연결·주말 플래그는 안 씀 | 40에폭, hidden 64, 조기 종료 10 |
| ST-GCN | `stgcn` | ST-GCN 블록 2개(시간 합성곱 → 그래프 합성곱 → 시간 합성곱), 9채널 모두 사용 | 40에폭, hidden 64, 조기 종료 10 |
| ST-GCN + Segment | `stgcn_segment` | ST-GCN 예측에 세그먼트 단위 보정값(NetVLAD 풀링 + Cluster-LSTM)을 더함 | 30에폭, hidden_lstm 256, 조기 종료 없음 |

```bash
python scripts/train.py lstm
python scripts/train.py stgcn --epochs 80 --lr 1e-3   # 모델별 기본값 덮어쓰기
```

모델별 기본값은 `traffic/training.py`의 `DEFAULTS`에 있습니다. 주요 옵션:

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `model` (위치 인자) | — | `lstm`, `stgcn`, `stgcn_segment` 중 하나 |
| `--epochs` | 모델별 설정 | 에폭 수 |
| `--batch` / `--lr` / `--clip` | `8` / `5e-4` / `5.0` | 배치 크기, 학습률 (Adam), gradient clipping (0이면 끔) |
| `--save_freq` | `5` | N 에폭마다 체크포인트 저장 (검증 MSE가 좋아진 에폭도 저장) |
| `--patience` | 모델별 설정 | 검증 MSE가 N 에폭 동안 좋아지지 않으면 조기 종료 |
| `--sched` | `plateau` | LR 스케줄러 (`plateau`: 검증 MSE가 정체되면 절반, `cosine`) |
| `--workers` | 자동 | DataLoader 작업자 수 (0이면 메인 프로세스에서 읽음) |
| `--hidden_dim` / `--num_layers` / `--dropout` | `64` / `1` / `0` | lstm 전용 |
| `--hidden` | `64` | stgcn 전용 |
| `--hidden_lstm` / `--horizon` / `--cluster_map` | `256` / `1` / processed 매핑 | stgcn_segment 전용 |

모델에 없는 옵션을 주면 에러로 멈춥니다 (예: `train.py lstm --hidden 32`).

## 평가

```bash
python scripts/evaluate.py lstm
python scripts/evaluate.py stgcn --ckpt outputs/stgcn/checkpoints/stgcn_ep040.pt
```

`--ckpt`를 생략하면 마지막 에폭의 체크포인트를 씁니다. 그 밖의 옵션은 `--batch`(기본 8), `--workers`, `--dpi`(기본 150)입니다.

## 산출물

```text
outputs/{모델명}/
├── checkpoints/{모델명}_epNNN.pt   # train.py: 5에폭마다 + 검증 MSE가 좋아진 에폭
├── metrics.csv                     # train.py: 에폭별 학습·검증 MSE, MAE
├── training_curve.png              # evaluate.py: 학습·검증 손실 곡선
├── test_metrics.csv                # evaluate.py: 테스트셋 MAE, RMSE, R² (정규화 값 기준)
└── figures/
    └── actual_vs_pred_{car,bus,truck,motorcycle}.png   # evaluate.py: 차종별 실제 vs 예측 대기 차량 수
```

`actual_vs_pred_*.png`는 테스트셋 예측을 실제 단위로 되돌려 차로 합계를 낸 뒤, 하루 시간대(5분)별로 평균한 그래프입니다.

## 결과

테스트셋, 40에폭 체크포인트, 정규화(z-score) 값 기준입니다 (8채널 전체로 계산).

| 모델 | MAE | RMSE | R² |
|---|---|---|---|
| LSTM | 0.3607 | 0.9894 | 0.0647 |
| ST-GCN | 0.2774 | 0.8556 | 0.3006 |
| ST-GCN + Segment | — | — | — (아래 1번) |

## 알려진 문제

1. **ST-GCN + Segment의 정답 누설.** 학습·평가 코드가 입력 1시간 중 가장 오래된 시점을 정답으로 쓰고, 그 정답을
   `residual_seq`로 모델에 그대로 넣습니다. 미래 예측이 아니라서 다른 모델과 비교할 수 없으므로 `evaluate.py`는
   이 모델의 학습 곡선만 그립니다. 예전에 학습한 결과(`outputs/stgcn_segment/`, git 제외)는 `--hidden_lstm 64`,
   40에폭으로 학습한 것이며 설명은 같은 폴더의 `NOTE.txt`에 있습니다.
2. **대기 차량 수의 15분 주기.** 매 15분의 마지막 5분(:10, :25, :40, :55 슬롯)의 대기 차량 수가 나머지 두 슬롯의
   약 2.7배입니다 (31일 합계). 원본 od 데이터에서 온 패턴이라 실제 vs 예측 그래프의 실제 값이 15분 간격 톱니 모양입니다.
3. **학습이 덜 됐을 수 있습니다.** LSTM과 ST-GCN 모두 조기 종료 없이 40에폭을 다 돌았고, 마지막 40에폭에서
   검증 MSE가 가장 낮았습니다. 에폭을 늘리면 더 좋아질 수 있습니다.
4. **재현성.** 같은 환경에서는 전처리 결과가 항상 같지만, CPU·라이브러리가 다르면 정규화 텐서와 학습 입력의
   float32 마지막 자리(최대 1.5e-5)가 달라질 수 있습니다.

## 점검 도구

```bash
python scripts/check/check_models.py                  # 세 모델 forward (더미 입력)
python scripts/check/check_windows.py                 # 학습 입력 shape, NaN, 연속성
python scripts/check/check_data_loader.py             # 분할 크기, 배치 shape, 정규화
python scripts/check/check_graph.py                   # 인접행렬 노드·엣지 수, 그래프 그림
python scripts/check/check_segments.py                # 세그먼트별 차로 수
python scripts/check/check_tensors.py                 # 날짜별 텐서 통계 CSV
python scripts/check/check_parquet.py data/raw/turn.parquet   # Parquet 메타데이터·샘플
```

## (선택) SUMO 시뮬레이션 입력

```bash
python scripts/build_sumo.py                          # → processed/sumo/<날짜>/ (31일 약 4분)
python scripts/build_sumo.py --dates 20220810
```

회전 교통량으로 만든 차량 흐름, 차로별 검지기, 신호 현시, 도로망 복사본과 `bucheon.sumocfg`를 날짜별로 만듭니다.
학습·평가에는 쓰지 않습니다.
