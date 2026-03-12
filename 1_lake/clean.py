# src/etl/rewrite_signal_xml.py
import os
import dask.dataframe as dd
from dask.diagnostics import ProgressBar

# 원본/목표 디렉토리
SRC = os.path.join('1_lake', 'indicator.parquet')
DST = os.path.join('1_lake', 'indicator_clean.parquet')

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