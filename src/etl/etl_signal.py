#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
etl_signal_parquet.py

부천시 신호데이터(CSV, XML) → Parquet
• Mac 리소스포크(. _*) 자동 필터링
• 제어문자·쓰레기값 없음
• 원본 필드 100% 보존
• 연구에 맞는 타입 캐스팅
usage: python etl_signal_parquet.py
"""
from __future__ import annotations
import re, glob, json
from pathlib import Path

import pandas as pd
import xmltodict
import dask.dataframe as dd
from dask import delayed
from dask.diagnostics import ProgressBar

RAW_DIR     = "0_raw/신호데이터"
OUT_DIR_CSV = "data/raw/signal_csv.parquet"
OUT_DIR_XML = "data/raw/signal_xml.parquet"

# ────────────────────────────────────────────────────────
# 1) 전체 파일 목록 수집 & Mac 리소스포크(. _*) 필터링
# ────────────────────────────────────────────────────────
all_csv = glob.glob(f"{RAW_DIR}/**/csv/*/*.csv", recursive=True)
all_xml = glob.glob(f"{RAW_DIR}/**/xml/*/*.xml", recursive=True)

def is_real_file(p: str, ext: str) -> bool:
    name = Path(p).name
    return not name.startswith("._") and name.lower().endswith(ext)

csv_paths = [p for p in all_csv if is_real_file(p, ".csv")]
xml_paths = [p for p in all_xml if is_real_file(p, ".xml")]

print(f"🔍 CSV 파일 수: {len(csv_paths):,}")
print(f"🔍 XML  파일 수: {len(xml_paths):,}")

# ────────────────────────────────────────────────────────
# 2) 파일명에서 메타 추출용 정규식
# ────────────────────────────────────────────────────────
VEH_RE_CSV = re.compile(r"([pbstlmk])\.csv$", re.I)
VEH_RE_XML = re.compile(r"([pbstlmk])\.xml$", re.I)
TS_RE_XML  = re.compile(r"(\d{14})[pbstlmk]\.xml$", re.I)

# ────────────────────────────────────────────────────────
# 3) CSV → pandas.DataFrame
# ────────────────────────────────────────────────────────
def parse_one_csv(fp: str) -> pd.DataFrame:
    p = Path(fp)
    # --- 메타
    route    = p.parents[2].name       # .../교차로_XX/csv/YYYYMMDD/file.csv
    date     = int(p.parents[0].name)  # "YYYYMMDD" → 20220810
    m        = VEH_RE_CSV.search(p.name)
    veh_type = m.group(1).lower() if m else "unknown"

    # --- 원본 읽기 (모두 문자열로)
    df = pd.read_csv(fp, dtype=str)

    # --- phasepattern 컬럼이 없으면 빈값 채우기
    if "phasepattern" not in df.columns:
        df["phasepattern"] = pd.NA

    # --- 메타 칼럼 추가
    df["route"]    = route
    df["date"]     = date
    df["veh_type"] = veh_type
    df["path"]     = str(p)

    # --- 타입 캐스팅
    df = df.astype({
        "interid"        : "string",
        "aringstarttime" : "string",
        "signalstate"    : "string",
        "phasepattern"   : "string",
        "route"          : "string",
        "veh_type"       : "string",
        "path"           : "string",
    })
    df["unix_time"] = pd.to_numeric(df["unix_time"], errors="coerce").astype("Int64")
    df["date"]      = df["date"].astype("Int32")

    # --- dt: unix_time → 15분 버킷, datetime64[ns]
    df["dt"] = (
        pd.to_datetime(df["unix_time"], unit="s", errors="coerce")
          .dt.floor("15min")
          .astype("datetime64[ns]")
    )
    return df

# ────────────────────────────────────────────────────────
# 4) XML → pandas.DataFrame
# ────────────────────────────────────────────────────────
def parse_one_xml(fp: str) -> pd.DataFrame:
    p = Path(fp)
    # --- 메타
    route    = p.parents[2].name
    date     = int(p.parents[0].name)
    m        = VEH_RE_XML.search(p.name)
    veh_type = m.group(1).lower() if m else "unknown"
    # 타임스탬프(YYYYMMDDHHMMSS) → dt_floor
    ts_m     = TS_RE_XML.search(p.name)
    base_dt  = pd.to_datetime(ts_m.group(1), format="%Y%m%d%H%M%S") if ts_m else pd.NaT

    # --- XML 파싱
    doc      = xmltodict.parse(p.read_text(encoding="utf-8"))
    tplogics = doc.get("tlLogics", {}).get("tlLogic", [])
    if isinstance(tplogics, dict):
        tplogics = [tplogics]

    rows: list[dict] = []
    for tl in tplogics:
        offset    = tl.get("@offset")
        prog_id   = tl.get("@programID")
        typ       = tl.get("@type")
        logic_id  = tl.get("@id")
        phases    = tl.get("phase", [])
        if isinstance(phases, dict):
            phases = [phases]
        for ph in phases:
            rows.append({
                "offset"         : offset,
                "programID"      : prog_id,
                "type"           : typ,
                "id"             : logic_id,
                "phase_duration" : ph.get("@duration"),
                "phase_state"    : ph.get("@state"),
                "route"          : route,
                "date"           : date,
                "veh_type"       : veh_type,
                "path"           : str(p),
                "dt"             : base_dt.floor("15min"),
            })

    df = pd.DataFrame(rows)

    # --- 타입 캐스팅
    df = df.astype({
        "offset"         : "string",
        "programID"      : "Int64",
        "type"           : "string",
        "id"             : "string",
        "phase_duration" : "Int64",
        "phase_state"    : "string",
        "route"          : "string",
        "date"           : "Int32",
        "veh_type"       : "string",
        "path"           : "string",
        "dt"             : "datetime64[ns]",
    })
    return df

# ────────────────────────────────────────────────────────
# 5) Dask → Parquet 저장 공통 함수
# ────────────────────────────────────────────────────────
def to_parquet(delayed_fn, paths: list[str], meta: dict, out_dir: str):
    tasks = [delayed(delayed_fn)(fp) for fp in paths]
    ddf   = dd.from_delayed(tasks, meta=meta)
    ProgressBar().register()
    ddf.to_parquet(
        out_dir,
        engine="pyarrow",
        compression="zstd",
        partition_on=["route","date","veh_type"],
        write_index=False,
    )
    print(f"✓ Parquet 저장 완료 → {out_dir}")
    print("■ 저장된 칼럼:")
    print(json.dumps(list(meta.keys()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    meta_csv = {
        "interid": "string",
        "aringstarttime": "string",
        "signalstate": "string",
        "unix_time": "Int64",        # move unix_time before phasepattern
        "phasepattern": "string",
        "route": "string",
        "date": "Int32",
        "veh_type": "string",
        "path": "string",
        "dt": "datetime64[ns]",
    }
    meta_xml = {
        "offset":"string","programID":"Int64","type":"string","id":"string",
        "phase_duration":"Int64","phase_state":"string",
        "route":"string","date":"Int32","veh_type":"string",
        "path":"string","dt":"datetime64[ns]",
    }

    # 5-1) CSV → Parquet
    to_parquet(
        parse_one_csv, csv_paths, meta_csv, OUT_DIR_CSV
    )
    # 5-2) XML → Parquet
    to_parquet(
        parse_one_xml, xml_paths, meta_xml, OUT_DIR_XML
    )