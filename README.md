# 차로 단위 교통 예측 프로젝트

부천시 교차로 주변 1,370개 차로의 교통 상태를 5분 단위로 예측하는 프로젝트입니다.
과거 1시간(5분 × 12스텝)의 차종별 대기 차량 수·평균 속도를 보고 다음 5분을 예측하며,
LSTM, ST-GCN, ST-GCN + Segment 세 모델을 하나씩 학습·비교합니다.

## 요구 사항

| 항목 | 버전 |
|---|---|
| Python | 3.13 |
| PyTorch | 2.13.0+cu126 |
| GPU | 선택 (CUDA가 있으면 자동 사용, 없으면 CPU) |

## 설치

```bash
pip install -r requirements.txt
```

`traffic/`이 저장소 루트에 있어 루트에서 `import traffic`이 바로 됩니다 (별도 패키지 설치 불필요).

> **torch는 `+cu126` 휠로 고정돼 있습니다** (CUDA 12.x 드라이버용). 다른 드라이버면 `nvidia-smi`로 확인해
> `requirements.txt`의 index-url과 태그를 바꾸세요.

## 빠른 시작

```bash
# 1. 데이터를 data/raw/에 배치 (아래 "데이터 준비" 참고)

# 2. 전처리: data/raw → processed/ (약 1분)
python scripts/build_dataset.py

# 3. 학습 → 평가 (모델 하나씩)
python scripts/train.py stgcn
python scripts/evaluate.py stgcn
```

## 프로젝트 구조

```text
.
├── data/raw/                      # 원본 데이터 (git 제외)
├── processed/                     # 전처리 결과 (git 제외, build_dataset.py로 다시 만듦)
├── outputs/{모델명}/              # 학습·평가 결과 (git 제외)
│
├── scripts/                       # 진입점 (전부 저장소 루트에서 실행)
│   ├── build_dataset.py           # 전처리 data/raw → processed/
│   ├── train.py                   # 학습 (train.py <모델명>)
│   ├── evaluate.py                # 평가 (evaluate.py <모델명>)
│   ├── etl/                       # AI Hub 원본 → data/raw 변환 (기록용)
│   └── check/                     # 점검 도구 (화면 출력만)
│
├── traffic/                       # 라이브러리 (설치 불필요, 루트에서 import)
│   ├── config.py                  # ★ 경로, 날짜 31일, 상수
│   ├── preprocess/                # 전처리 4단계 (graph, segments, tensors, windows)
│   ├── data.py                    # Dataset, DataLoader (시간 순 70/15/15 분할)
│   ├── models/                    # lstm.py, stgcn.py, stgcn_segment.py
│   ├── training.py                # 공통 학습 루프, 모델별 기본값 (DEFAULTS)
│   └── evaluation.py              # 테스트 지표, 그래프
│
├── notebooks/build_tensor.ipynb   # 텐서 만드는 과정을 하루치로 따라가는 노트북
├── requirements.txt
└── README.md
```

## 데이터 준비

[Google Drive](GOOGLE_DRIVE_LINK)에서 받아 `data/raw/`에 둡니다 (약 3.5GB):

```text
data/raw/
├── od.parquet/            # 기종점 교통량: 차량별 시각·차로·차종·속도·대기 여부   ← 학습에 사용
├── net/                   # SUMO 도로망 bucheon.net.xml                           ← 학습에 사용
├── turn.parquet/          # 회전 교통량
├── signal_xml.parquet/    # 신호 현시 계획
├── signal_csv.parquet/    # 신호 상태 기록
└── indicator.parquet/     # 구간별 평균 속도·지체 시간
```

[AI Hub](https://www.aihub.or.kr)의 「AI 기반 신호 최적화를 위한 데이터」(2-058) 중 부천시 2022년 8~10월 31일치를
Parquet으로 변환한 것입니다. 변환 코드는 `scripts/etl/`에 있습니다.

## 전처리

```bash
python scripts/build_dataset.py                    # 4단계 전부 (약 1분)
python scripts/build_dataset.py tensors windows    # 일부 단계만 (뒤 단계는 앞 단계 결과를 읽음)
```

| 단계 | 하는 일 |
|---|---|
| graph | 도로망에서 차로 1,370개와 차로 사이 연결을 읽어 정규화된 인접행렬을 만듦 |
| segments | 도로를 5개씩 묶어 차로마다 세그먼트 번호(88개)를 매김 |
| tensors | od 기록을 날짜 × 5분 슬롯 × 차로 × 8채널로 집계하고, 날짜·채널별 z-score로 정규화 |
| windows | 하루 안에서 과거 12스텝 입력과 다음 5분 정답을 잘라 학습 샘플을 만들고, 주말·공휴일 플래그를 붙임 |

8채널은 차종 4종(승용, 버스, 트럭, 오토바이)별 대기 차량 수와 평균 속도입니다. 생성 파일:

```text
processed/
├── graph/
│   ├── lanes.txt          # 차로 ID 1,370개 (모든 배열의 차로 순서)
│   ├── adjacency.npy      # 정규화된 인접행렬 (1370, 1370)
│   └── segments.npy       # 차로별 세그먼트 번호 (1370,)
├── tensors/
│   ├── values.npy         # (31일, 288 슬롯, 1370 차로, 8채널) 실제 값
│   ├── normalized.npy     # (31, 288, 1370, 8) 날짜·채널별 z-score
│   └── dates.npy          # (31,)
└── windows/
    ├── X.npy              # (8556, 12, 1370, 9) 입력: 8채널 + 주말·공휴일 플래그
    ├── Y.npy              # (8556, 1370, 8) 정답: 다음 5분
    └── dates.npy          # (8556,) 윈도우 날짜
```

윈도우는 시간 순서대로 학습·검증·테스트 70/15/15 (5,989 / 1,283 / 1,284개)로 나눕니다.
테스트는 2022-10-26 09:00 ~ 10-31 23:55 (예측 시점 기준)입니다.

## 실행 방법

모델은 이름으로 하나씩 학습합니다. 모델별 기본값은 `traffic/training.py`의 `DEFAULTS`에 있습니다.

```bash
python scripts/train.py lstm
python scripts/train.py stgcn --epochs 80 --lr 1e-3   # 모델별 기본값 덮어쓰기
```

주요 옵션:

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `model` (위치 인자) | — | 학습할 모델 하나 (아래 3개 중) |
| `--epochs` | 모델별 설정 | 에폭 수 |
| `--batch` | `8` | 배치 크기 |
| `--lr` | `5e-4` | 학습률 (Adam, weight decay 1e-4) |
| `--clip` | `5.0` | gradient clipping (0이면 끔) |
| `--save_freq` | `5` | N 에폭마다 체크포인트 저장 (검증 MSE가 좋아진 에폭도 저장) |
| `--patience` | 모델별 설정 | 검증 MSE가 N 에폭 동안 좋아지지 않으면 조기 종료 |
| `--sched` | `plateau` | LR 스케줄러 (`plateau`, `cosine`) |
| `--workers` | 자동 | DataLoader 작업자 수 (0이면 메인 프로세스) |

모델에 없는 옵션을 주면 에러로 멈춥니다 (예: `train.py lstm --hidden 32`).

### 학습 가능한 모델 3개

| 모델 | 파라미터 | 구조 | 기본 설정 | 전용 옵션 |
|---|---|---|---|---|
| `lstm` | 19K | 차로마다 같은 LSTM으로 과거 1시간을 봄 (도로 연결·플래그 안 씀) | 40에폭, 조기 종료 10 | `--hidden_dim` `--num_layers` `--dropout` |
| `stgcn` | 54K | 시간 합성곱 + 그래프 합성곱 블록 2개 | 40에폭, 조기 종료 10 | `--hidden` |
| `stgcn_segment` | 586K | ST-GCN 예측에 세그먼트별 보정값(NetVLAD + Cluster-LSTM)을 더함 | 30에폭, 조기 종료 없음 | `--hidden_lstm` `--horizon` `--cluster_map` |

> `stgcn_segment`는 학습 코드에 정답 누설이 있어 평가에서 제외합니다 (`evaluate.py`는 학습 곡선만 그림).

## 결과 분석

```bash
python scripts/evaluate.py stgcn                                                   # 마지막 에폭 체크포인트
python scripts/evaluate.py stgcn --ckpt outputs/stgcn/checkpoints/stgcn_ep040.pt   # 체크포인트 지정
```

옵션: `--ckpt` `--batch` `--workers` `--dpi`

## 산출물

```text
outputs/{모델명}/
├── checkpoints/{모델명}_epNNN.pt   # train.py: 5에폭마다 + 검증 MSE가 좋아진 에폭
├── metrics.csv                     # train.py: 에폭별 학습·검증 MSE, MAE
├── training_curve.png              # evaluate.py: 학습 곡선
├── test_metrics.csv                # evaluate.py: 테스트셋 MAE, RMSE, R²
└── figures/
    └── actual_vs_pred_{car,bus,truck,motorcycle}.png   # evaluate.py: 차종별 실제 vs 예측 대기 차량 수
```

`actual_vs_pred_*.png`는 테스트셋 예측을 실제 단위로 되돌려 차로 합계를 낸 뒤, 하루 시간대(5분)별로 평균한 그래프입니다.

## 점검 도구

```bash
python scripts/check/check_models.py          # 세 모델 forward (더미 입력)
python scripts/check/check_graph.py           # 차로·링크 수, 연결 요소, 그래프 그림 (창)
python scripts/check/check_segments.py        # 세그먼트별 차로 수
python scripts/check/check_tensors.py         # 텐서 NaN, 채널별 통계, 정규화 확인
python scripts/check/check_windows.py         # 학습 샘플 shape, NaN, 플래그, 연속성
python scripts/check/check_data_loader.py     # 분할 크기, 배치 shape, 정규화
python scripts/check/check_parquet.py data/raw/turn.parquet   # Parquet 메타데이터·샘플
```
