# scripts/etl/clean_indicator.py
# data/raw/indicator.parquet에서 veh_type 컬럼을 빼고 route·date 파티션으로 다시 저장합니다.
import sys
from pathlib import Path

import dask.dataframe as dd
from dask.diagnostics import ProgressBar

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from traffic import config

# 원본/목표 디렉토리
SRC = str(config.RAW_DIR / 'indicator.parquet')
DST = str(config.PROCESSED_DIR / 'indicator_clean.parquet')

# 1) Parquet 로드(모든 파티션 읽음)
df = dd.read_parquet(SRC)

# 2) veh_type 컬럼 삭제
df = df.drop(columns=['veh_type'])

# 3) 재파티셔닝: date, route 만 남깁니다.
with ProgressBar():
    df.to_parquet(
        DST,
        engine='pyarrow',
        write_index=False,
        partition_on=['route', 'date']
    )

print(f"Rewrote parquet without veh_type into {DST}")
