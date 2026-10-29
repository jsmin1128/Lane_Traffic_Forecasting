#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_dataset.py

data/raw → processed/ 전처리를 순서대로 실행합니다 (전체 약 5분).

  1) graph     도로망 → 차로 목록 lanes.txt, 정규화된 인접행렬 A_lane.npy        (processed/graph)
  2) segments  차로 → 세그먼트 매핑 lane_to_segment_id_by_edge.npy               (processed/graph)
  3) tensors   od.parquet → 날짜별 input_tensor·normalized_tensor              (processed/tensor)
  4) flags     정규화 텐서 + 주말·공휴일 플래그 → normalized_tensor_*_with_flag.npy (processed/tensor)
  5) windows   학습 입력 all_X.npy, all_Y.npy, all_DATE.npy                     (processed/windows)

Usage (저장소 루트에서 실행):
    python scripts/build_dataset.py                    # 전부
    python scripts/build_dataset.py flags windows      # 일부 단계만
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from traffic import config
from traffic.preprocess import graph, segments, tensors, windows

STEPS = ['graph', 'segments', 'tensors', 'flags', 'windows']


def parse_args():
    p = argparse.ArgumentParser(description="Build processed/ from data/raw")
    p.add_argument('steps', nargs='*', help=f"실행할 단계 {STEPS} (기본: 전부)")
    p.add_argument('--raw_dir',       type=Path, default=config.RAW_DIR,       help='원본 폴더')
    p.add_argument('--processed_dir', type=Path, default=config.PROCESSED_DIR, help='결과 폴더')
    args = p.parse_args()
    unknown = [s for s in args.steps if s not in STEPS]
    if unknown:
        p.error(f"알 수 없는 단계: {unknown} (가능: {STEPS})")
    return args


def main():
    args = parse_args()
    steps = args.steps or STEPS
    net_xml    = args.raw_dir / 'net' / 'bucheon.net.xml'
    graph_dir  = args.processed_dir / 'graph'
    tensor_dir = args.processed_dir / 'tensor'

    if 'graph' in steps:
        graph.build_graph(net_xml, graph_dir)
    if 'segments' in steps:
        segments.build_segments(net_xml, graph_dir / 'lanes.txt',
                                graph_dir / 'lane_to_segment_id_by_edge.npy', config.SEGMENT_GROUP_SIZE)
    if 'tensors' in steps:
        tensors.build_tensors(args.raw_dir / 'od.parquet', graph_dir / 'lanes.txt', tensor_dir, config.DATES)
    if 'flags' in steps:
        tensors.add_flags(tensor_dir, config.DATES)
    if 'windows' in steps:
        windows.make_windows(tensor_dir, args.processed_dir / 'windows', config.DATES)


if __name__ == '__main__':
    main()
