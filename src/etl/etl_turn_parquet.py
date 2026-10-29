#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
부천시 회전교통량 XML ➜ Parquet (원본 필드 100 % 보존)
usage: python etl_turn_parquet.py
"""
from __future__ import annotations
import re
import glob
import json
from pathlib import Path

import xmltodict
import pandas as pd
import dask.dataframe as dd
from dask import delayed
from dask.diagnostics import ProgressBar
import dask

# 원본 디렉토리, 출력 디렉토리
RAW_DIR = Path("0_raw/교통량데이터(회전교통량기반)")
OUT_DIR = Path("data/raw/turn.parquet")

# Mac 리소스 포크나 숨김 파일 필터링: 실제 xml 파일만
all_paths = glob.glob(str(RAW_DIR / "**" / "*.xml"), recursive=True)
xml_paths = [p for p in all_paths
             if not Path(p).name.startswith("._")
             and Path(p).suffix.lower() == ".xml"]
print(f"🔍 처리 대상 XML 파일 수: {len(xml_paths)}")

# 차량 타입 추출 정규표현식
VEH_RE = re.compile(r"([pbstlmk])(?:tra|total)?\.xml$", re.I)

# XML 하나를 pandas DataFrame으로 변환

def parse_one_xml(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    # 파일 메타
    route = path.parents[1].name  # 예: 교차로_26
    date = path.parents[0].name   # 예: 20220810
    m = VEH_RE.search(path.name)
    veh_type = m.group(1).lower() if m else "unknown"

    # XML 파싱
    raw = path.read_text(encoding="utf-8")
    # resource fork나 제어문자 제거
    raw = re.sub(r"[\x00-\x1F]+", "", raw)
    doc = xmltodict.parse(raw)

    intervals = doc.get("data", {}).get("interval", [])
    if isinstance(intervals, dict):
        intervals = [intervals]

    rows: list[dict] = []
    for itv in intervals:
        begin = itv.get("@begin")
        end = itv.get("@end")
        itv_id = itv.get("@id")
        edges = itv.get("edgeRelation", [])
        if isinstance(edges, dict):
            edges = [edges]
        for e in edges:
            rows.append({
                # 원본 필드
                "begin": begin,
                "end": end,
                "interval_id": itv_id,
                "edge_from": e.get("@from"),
                "edge_to": e.get("@to"),
                "count": e.get("@count"),
                # 메타
                "route": route,
                "date": date,
                "veh_type": veh_type,
                "filepath": str(path),
            })

    df = pd.DataFrame(rows)

    # --- numeric type conversions for research-friendly schema ---
    df["begin"] = pd.to_numeric(df["begin"], errors="coerce").astype("float64")
    df["end"]   = pd.to_numeric(df["end"],   errors="coerce").astype("float64")
    df["date"]  = pd.to_numeric(df["date"],  errors="coerce").astype("Int32")

    # dtype 캐스팅
    str_cols = ["begin", "end", "interval_id",
                "edge_from", "edge_to",
                "route", "veh_type", "filepath"]
    for c in str_cols:
        df[c] = df[c].astype("string")
    df["count"] = pd.to_numeric(df["count"], errors="coerce").astype("Int32")
    # bucketed timestamp
    df["dt"] = (
        pd.to_datetime(df["begin"].astype("float64"), unit="s")
        .dt.floor("15min")
    )
    return df

# Dask 설정: threads 스케줄러, ProgressBar
if __name__ == "__main__":
    dask.config.set(scheduler="threads")
    ProgressBar().register()

    # delayed list
    delayed_dfs = [delayed(parse_one_xml)(p) for p in xml_paths]

    # meta 정의
    meta = {
        "begin": "string",
        "end": "string",
        "interval_id": "string",
        "edge_from": "string",
        "edge_to": "string",
        "count": "Int32",
        "route": "string",
        "date": "Int32",
        "veh_type": "string",
        "filepath": "string",
        "dt": "datetime64[ns]",
    }
    ddf = dd.from_delayed(delayed_dfs, meta=meta)
    # repartition to a reasonable number for parallel writes
    ddf = ddf.repartition(npartitions=1000)

    # Parquet 저장
    print("⏳ Parquet 저장 시작…")
    ddf.to_parquet(
        OUT_DIR,
        engine="pyarrow",
        compression="zstd",
        partition_on=["route", "date", "veh_type"],
        write_index=False,
    )
    print(f"✓ 완료 → {OUT_DIR}")

    # 저장 컬럼 확인
    print("\n■ 저장된 컬럼:")
    print(json.dumps(list(ddf.columns), ensure_ascii=False, indent=2))