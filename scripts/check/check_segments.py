#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
check_segments.py

processed/graph/lane_to_segment_id_by_edge.npy (차로 → 세그먼트 매핑)를 검증하고,
세그먼트별 차로 수 통계를 출력합니다.

사용법: python scripts/check/check_segments.py
"""

import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from traffic import config

# 1) 매핑 파일 로드
graph_dir = config.PROCESSED_DIR / 'graph'
mapping = np.load(graph_dir / 'lane_to_segment_id_by_edge.npy')
lanes   = (graph_dir / 'lanes.txt').read_text().splitlines()

assert mapping.ndim == 1, "매핑 배열이 일차원이 아닙니다."
assert mapping.shape[0] == len(lanes), "lanes.txt 길이와 매핑 길이 불일치"

# 2) 기본 통계 출력
print(f"Mapping shape: {mapping.shape}")
unique_ids = np.unique(mapping)
unique_ids = unique_ids[unique_ids >= 0]  # -1 제외
print(f"Unique segment count: {unique_ids.size}")
print(f"Min segment ID: {mapping.min()}  Max segment ID: {mapping.max()}")

# 3) 클러스터별 lane 개수 계산
#    np.bincount를 쓰려면 음수가 없어야 하므로 -1 제거
counts = np.bincount(mapping[mapping >= 0])
#   counts[i] = segment i에 할당된 lane 수

print("\nCluster size statistics:")
print(f"  Max lanes in one cluster : {counts.max()}")
print(f"  Min lanes in one cluster : {counts.min()}")
print(f"  Mean lanes per cluster   : {counts.mean():.2f}")

# 4) 클러스터별 개수 분포 (간단한 텍스트 버전)
print("\nCluster size distribution (size: count of clusters):")
dist = Counter(counts)
for size, num in sorted(dist.items()):
    print(f"  {size:3d} lanes  → {num:3d} clusters")

# 5) 일부 예시 출력
print("\n첫 5개 매핑 예시:")
for i in range(5):
    print(f"  {lanes[i]} -> {mapping[i]}")
