#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
etl_indicator.py

AI Hub 지표데이터/*.csv ➜ data/raw/indicator.parquet
  • 원본 필드 100% 보존
  • Mac 리소스포크(. _*) 자동 필터링
  • 제어문자/쓰레기값(<NA>/NaN) 자동 제거
  • 연구에 맞춘 적절한 dtype 캐스팅
usage: python scripts/etl/etl_indicator.py <AI Hub 지표데이터 폴더>
"""
from __future__ import annotations
import re, glob, json
from pathlib import Path

import pandas as pd
import dask.dataframe as dd
from dask import delayed
from dask.diagnostics import ProgressBar

import argparse
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from traffic import config

_parser = argparse.ArgumentParser(description='AI Hub 지표데이터 CSV → Parquet')
_parser.add_argument('raw_dir', help="AI Hub '지표데이터' 폴더")
_parser.add_argument('--out_dir', default=str(config.RAW_DIR / 'indicator.parquet'), help='출력 Parquet 폴더')
_args = _parser.parse_args()

RAW_DIR = _args.raw_dir
OUT_DIR = _args.out_dir

# — 1) 모든 CSV 경로 수집 & Mac 리소스포크(. _*) 필터링
all_csv = glob.glob(f"{RAW_DIR}/**/*.csv", recursive=True)
csv_paths = [
    p for p in all_csv
    if not Path(p).name.startswith("._")
       and p.lower().endswith("total.csv")
]
print(f"🔍 처리 대상 CSV 파일 수: {len(csv_paths):,}")

# 파일명에서 날짜·시간(dt) 추출용 정규식
FNAME_RE = re.compile(r"(\d{8})(\d{6})total\.csv$", re.IGNORECASE)

def _clean_str_series(s: pd.Series) -> pd.Series:
    """string dtype에서 제어문자 제거, invalid는 <NA> 처리"""
    return (
        s
        .astype("string")
        .str.replace(r"[\x00-\x1F]", "", regex=True)
        .replace("", pd.NA)
    )

# — 2) 하나의 CSV → pandas.DataFrame
def parse_one_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    route = path.parents[1].name       # ex) 교차로_26
    date_str = path.parents[0].name    # ex) 20220810

    # 파일명에서 dt 생성
    m = FNAME_RE.search(path.name)
    if not m:
        raise RuntimeError(f"unexpected filename: {path.name}")
    dt = pd.to_datetime(m.group(1) + m.group(2),
                        format="%Y%m%d%H%M%S")

    # ---- 원본 읽기 ----
    df = pd.read_csv(
        path,
        dtype={
            "edge_grp_id": "string",
            "avg_spd":       "string",
            "total_time_loss": "string",
            "avg_time_loss":   "string",
            "avg_travle_time": "string",
        },
        na_values=["", "NA", "NaN"],
        keep_default_na=True
    )

    # ---- 제어문자 제거 & 타입 변환 ----
    df["edge_grp_id"]      = _clean_str_series(df["edge_grp_id"])
    df["avg_spd"]          = pd.to_numeric(df["avg_spd"], errors="coerce")
    df["total_time_loss"]  = pd.to_numeric(df["total_time_loss"], errors="coerce")
    df["avg_time_loss"]    = pd.to_numeric(df["avg_time_loss"], errors="coerce")
    df["avg_travle_time"]  = pd.to_numeric(df["avg_travle_time"], errors="coerce")

    # ---- 메타 컬럼 추가 ----
    df["route"]    = route
    df["date"]     = int(date_str)
    df["dt"]       = dt
    df["veh_type"] = "X"   # 지표데이터는 고정

    # pandas dtypes 통일
    df["route"]    = df["route"].astype("string")
    df["veh_type"] = df["veh_type"].astype("category")
    df["date"]     = df["date"].astype("int32")

    # 컬럼 순서 보존
    cols = [
      "edge_grp_id",
      "avg_spd", "total_time_loss", "avg_time_loss", "avg_travle_time",
      "route", "date", "dt", "veh_type"
    ]
    return df[cols]

# — 3) Dask로 병렬 처리 & Parquet 저장 ----
delayed_dfs = [delayed(parse_one_csv)(p) for p in csv_paths]
meta = {
    "edge_grp_id":      "string",
    "avg_spd":          "float64",
    "total_time_loss":  "float64",
    "avg_time_loss":    "float64",
    "avg_travle_time":  "float64",
    "route":            "string",
    "date":             "int32",
    "dt":               "datetime64[ns]",
    "veh_type":         "category",
}
ddf = dd.from_delayed(delayed_dfs, meta=meta)

ProgressBar().register()
ddf.to_parquet(
    OUT_DIR,
    engine="pyarrow",
    compression="zstd",
    partition_on=["route", "date"],
    write_index=False,
)

print(f"\n✅ Parquet 저장 완료 → {OUT_DIR}\n")
print("■ 저장되는 컬럼:")
print(json.dumps(list(ddf.columns), ensure_ascii=False, indent=2))