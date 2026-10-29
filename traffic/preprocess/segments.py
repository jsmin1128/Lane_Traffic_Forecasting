"""
차로 → 세그먼트 번호를 만듭니다 → segments.npy (ST-GCN + Segment 모델이 씀)

  segments.npy  lanes.txt 순서의 int32 배열 (길이 N). 값 = 그 차로가 속한 세그먼트 번호 (0부터)

만드는 순서
  1) 이웃 도로: 도로 a가 끝나는 교차로에서 도로 b가 시작하면 a와 b를 서로 이웃으로 봅니다.
  2) 도로를 너비 우선 탐색(BFS)으로 방문하고, 방문한 순서대로 group_size개씩 끊어 세그먼트로 묶습니다.
     가까운 도로끼리 같은 세그먼트가 됩니다. 이웃은 ID 순으로 방문하므로 실행할 때마다 결과가 같습니다.
  3) 차로는 자기 도로의 세그먼트를 따릅니다 (SUMO 차로 ID = <도로 ID>_<차로 번호>).
  4) lanes.txt에 나오는 세그먼트만 남기고 0부터 번호를 다시 매깁니다.
"""

import math
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
from lxml import etree


def read_edges(net_xml):
    """net.xml → {도로 ID: (시작 교차로, 끝 교차로)} (교차로 내부 도로 제외)"""
    root = etree.parse(str(net_xml)).getroot()
    return {edge.get('id'): (edge.get('from'), edge.get('to'))
            for edge in root.iter('edge')
            if not edge.get('id').startswith(':') and edge.get('function') != 'internal'}


def bfs_order(first, neighbors, visited):
    """first와 이어진 도로들을 가까운 순서(BFS)로 방문한 목록"""
    order, queue = [], deque([first])
    visited.add(first)
    while queue:
        edge = queue.popleft()
        order.append(edge)
        for nb in sorted(neighbors[edge]):
            if nb not in visited:
                visited.add(nb)
                queue.append(nb)
    return order


def group_edges(edges, group_size):
    """{도로 ID: (시작, 끝)} → {도로 ID: 세그먼트 번호}"""
    # 이웃 도로: a의 끝 교차로 = b의 시작 교차로
    starts_at = defaultdict(list)
    for edge, (start, _) in edges.items():
        starts_at[start].append(edge)
    neighbors = defaultdict(set)
    for a, (_, end) in edges.items():
        for b in starts_at[end]:
            neighbors[a].add(b)
            neighbors[b].add(a)

    segment_of, visited, n_segments = {}, set(), 0
    for first in edges:
        if first in visited:
            continue
        order = bfs_order(first, neighbors, visited)    # 서로 이어진 도로 덩어리 하나
        for i, edge in enumerate(order):
            segment_of[edge] = n_segments + i // group_size
        n_segments += math.ceil(len(order) / group_size)
    return segment_of


def build_segments(net_xml, lanes_path, out_path, group_size):
    """net.xml + lanes.txt → out_path (차로별 세그먼트 번호)"""
    segment_of = group_edges(read_edges(net_xml), group_size)

    lanes = Path(lanes_path).read_text(encoding='utf-8').split()
    segments = [segment_of[lane.rsplit('_', 1)[0]] for lane in lanes]   # 차로 → 자기 도로의 세그먼트
    _, mapping = np.unique(segments, return_inverse=True)                # 쓰인 번호만 0부터 다시

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, mapping.astype(np.int32))
    sizes = np.bincount(mapping)
    print(f"▶ 세그먼트 {len(sizes)}개 (도로 {group_size}개씩, 세그먼트당 차로 {sizes.min()}~{sizes.max()}개) → {out_path}")
