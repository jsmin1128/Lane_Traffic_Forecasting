"""
차로 → 세그먼트 매핑(lane_to_segment_id_by_edge.npy)을 만듭니다. ST-GCN + Segment 모델이 씁니다.

SUMO .net.xml에서 edge 연결을 읽어, 이어진 edge들(연결 컴포넌트)을 방문 순서대로
최대 max_group_size개씩 묶어 세그먼트로 만들고, lanes.txt 순서대로 lane마다 세그먼트 ID를 매깁니다.
이웃 edge를 정렬된 순서로 방문하므로 실행할 때마다 같은 결과가 나옵니다.
"""

import xml.etree.ElementTree as ET
from collections import defaultdict, deque
from pathlib import Path

import numpy as np


def build_segments(net_xml, lanes_path, out_path, max_group_size):
    M = max_group_size

    # 1) .net.xml 파싱
    root = ET.parse(str(net_xml)).getroot()

    # 2) edge 정보 수집: id, from-node, to-node
    edge_info = {}
    for edge in root.findall('.//edge'):
        eid  = edge.get('id')
        src  = edge.get('from')
        dst  = edge.get('to')
        edge_info[eid] = (src, dst)

    # 3) 인접성 그래프 생성 (방향 고려)
    neighbors = defaultdict(set)
    for e1, (s1, d1) in edge_info.items():
        for e2, (s2, d2) in edge_info.items():
            if d1 == s2:
                neighbors[e1].add(e2)
                neighbors[e2].add(e1)

    # 4) 연결 컴포넌트별로 최대 M개씩 sub-group으로 나눔
    visited = set()
    segment_id = {}
    seg_counter = 0
    for eid in edge_info:
        if eid in visited:
            continue
        comp = []
        dq = deque([eid])
        visited.add(eid)
        while dq:
            cur = dq.popleft()
            comp.append(cur)
            for nb in sorted(neighbors[cur]):   # 정렬해서 방문 → 실행할 때마다 같은 결과
                if nb not in visited:
                    visited.add(nb)
                    dq.append(nb)
        # comp 내에서 M개씩 끊어서 ID 할당
        for i in range(0, len(comp), M):
            for e in comp[i:i+M]:
                segment_id[e] = seg_counter
            seg_counter += 1

    # 5) lanes.txt 읽고 lane→edge→segment 매핑
    with open(lanes_path) as f:
        lanes = [ln.strip() for ln in f if ln.strip()]

    mapping = np.zeros(len(lanes), dtype=np.int32)
    for i, lid in enumerate(lanes):
        base_edge = lid.rsplit('_', 1)[0]       # e.g. "edgeID_0" → "edgeID"
        mapping[i] = segment_id.get(base_edge, -1)

    # 6) 빈 클러스터(사용되지 않은 segment ID) 제거하고
    #    0 이상 ID만 0부터 연속된 ID로 재할당
    used_ids = sorted({sid for sid in mapping.tolist() if sid >= 0})
    new_id_map = {old: new for new, old in enumerate(used_ids)}
    mapping_compressed = np.array(
        [new_id_map[sid] if sid >= 0 else -1 for sid in mapping],
        dtype=np.int32
    )

    # 7) 결과 저장 및 요약 출력
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, mapping_compressed)
    print(f"저장 완료: '{out_path}'")
    print(f"총 lanes: {len(lanes)}")
    print(f"원본 segment 개수: {seg_counter}")
    print(f"사용된 segment (압축 후) 개수: {len(used_ids)}")
    print("예시 (first 5):", mapping_compressed[:5].tolist())
