#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_sumo.py

(선택) SUMO 시뮬레이션 입력 파일을 날짜별로 만듭니다 → processed/sumo/<날짜>/ (31일 약 4분)
학습·평가에는 쓰지 않고, SUMO로 시뮬레이션할 때만 필요합니다.

Usage (저장소 루트에서 실행):
    python scripts/build_sumo.py
    python scripts/build_sumo.py --dates 20220810 20220811
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from traffic import config
from traffic.preprocess.sumo import build_sumo


def main():
    p = argparse.ArgumentParser(description="Build SUMO input files per date")
    p.add_argument('--dates',   type=int, nargs='+', default=config.DATES, help='만들 날짜 (기본: 31일 전부)')
    p.add_argument('--raw_dir', type=Path, default=config.RAW_DIR)
    p.add_argument('--out_dir', type=Path, default=config.PROCESSED_DIR / 'sumo')
    args = p.parse_args()
    build_sumo(args.raw_dir, args.out_dir, args.dates)


if __name__ == '__main__':
    main()
