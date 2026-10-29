"""
도로망(SUMO net.xml)으로 차로 그래프를 만듭니다 → <out_dir>/lanes.txt, adjacency.npy

  lanes.txt      그래프의 노드 = 차로 ID 목록. 이 순서가 텐서와 모델의 차로 순서입니다.
  adjacency.npy  정규화된 인접행렬 A (N×N, float32). A[i, j] > 0 이면 차로 i에서 차로 j로 갈 수 있습니다.

만드는 순서
  1) 노드: 일반 도로(edge)의 차로. 교차로 내부 도로(id가 ':'로 시작)는 뺍니다.
  2) 링크: <connection from="도로 a" to="도로 b">마다 a의 모든 차로 → b의 모든 차로에 +1
           (방향 있음, 차로별 회전 방향은 구분하지 않음)
  3) 링크가 하나도 없는 고립 차로는 뺍니다.
  4) 대각선(자기 자신 연결)을 1로 채우고 D^-1/2 · A · D^-1/2 로 정규화합니다 (D = 행 합).
"""

from collections import Counter
from pathlib import Path

import numpy as np
from lxml import etree


def read_lanes_and_links(net_xml):
    """net.xml → (차로 ID 목록, {(출발 차로, 도착 차로): 연결 수})"""
    root = etree.parse(str(net_xml)).getroot()

    # 도로 ID → 그 도로의 차로 ID들 (교차로 내부 도로 제외)
    edge_lanes = {}
    for edge in root.iter('edge'):
        if edge.get('id').startswith(':') or edge.get('function') == 'internal':
            continue
        edge_lanes[edge.get('id')] = [lane.get('id') for lane in edge.findall('lane')]
    lanes = [lane for lane_ids in edge_lanes.values() for lane in lane_ids]

    # 도로 a → 도로 b 연결마다 a의 모든 차로 → b의 모든 차로에 +1
    links = Counter()
    for conn in root.iter('connection'):
        a, b = conn.get('from'), conn.get('to')
        if a in edge_lanes and b in edge_lanes:
            for src in edge_lanes[a]:
                for dst in edge_lanes[b]:
                    links[src, dst] += 1
    return lanes, links


def normalized_adjacency(lanes, links):
    """lanes 순서로 N×N 인접행렬을 채운 뒤 D^-1/2 · A · D^-1/2 로 정규화합니다."""
    index = {lane: i for i, lane in enumerate(lanes)}
    A = np.zeros((len(lanes), len(lanes)), dtype=np.float32)
    for (src, dst), count in links.items():
        A[index[src], index[dst]] = count   # A[출발, 도착]
    np.fill_diagonal(A, 1.0)                # 자기 자신 연결

    d = 1.0 / np.sqrt(A.sum(axis=1))        # D^-1/2 (대각선이 1이라 행 합은 1 이상)
    return d[:, None] * A * d[None, :]


def build_graph(net_xml, out_dir):
    """net.xml → out_dir/lanes.txt, adjacency.npy"""
    lanes, links = read_lanes_and_links(net_xml)

    # 고립 차로(링크가 하나도 없는 차로) 빼기
    linked = {lane for pair in links for lane in pair}
    isolated = [lane for lane in lanes if lane not in linked]
    lanes = [lane for lane in lanes if lane in linked]

    A_norm = normalized_adjacency(lanes, links)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'lanes.txt').write_text(''.join(f'{lane}\n' for lane in lanes), encoding='utf-8')
    np.save(out_dir / 'adjacency.npy', A_norm)
    print(f"▶ 차로 {len(lanes)}개 (고립 차로 {len(isolated)}개 제외), 링크 {len(links)}개 → {out_dir}/lanes.txt, adjacency.npy")
