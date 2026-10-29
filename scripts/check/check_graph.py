#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
check_graph.py

processed/graph의 lanes.txt와 adjacency.npy를 점검하고 그래프를 창으로 보여 줍니다 (파일은 만들지 않음).
  - 차로(노드) 수, 링크 수(자기 자신 연결 제외), 약한 연결 요소 수, 고립 차로 수
  - lanes.txt와 도로망(net.xml)의 차로 ID가 서로 맞는지
  - 도로망 좌표에 차로를 찍은 그래프 그림

사용법: python scripts/check/check_graph.py
"""

import argparse
import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from lxml import etree

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from traffic import config


def lane_positions(net_xml):
    """일반 도로의 차로마다 shape 좌표의 평균 위치"""
    pos = {}
    for edge in etree.parse(str(net_xml)).getroot().iter('edge'):
        if edge.get('id').startswith(':') or edge.get('function') == 'internal':
            continue
        for lane in edge.findall('lane'):
            points = [tuple(map(float, p.split(','))) for p in lane.get('shape', '').split()]
            if points:
                pos[lane.get('id')] = tuple(np.mean(points, axis=0))
    return pos


def main():
    p = argparse.ArgumentParser(description="Check the lane graph")
    p.add_argument('--net',   type=Path, default=config.RAW_DIR / 'net' / 'bucheon.net.xml')
    p.add_argument('--graph', type=Path, default=config.PROCESSED_DIR / 'graph', help='lanes.txt, adjacency.npy가 있는 폴더')
    args = p.parse_args()

    A = np.load(args.graph / 'adjacency.npy')
    lanes = (args.graph / 'lanes.txt').read_text(encoding='utf-8').split()
    assert A.shape == (len(lanes), len(lanes)), f"adjacency {A.shape}와 lanes.txt {len(lanes)}개가 맞지 않습니다."

    links = (A > 0) & ~np.eye(len(lanes), dtype=bool)          # 자기 자신 연결 제외
    G = nx.DiGraph()
    G.add_nodes_from(lanes)
    G.add_edges_from((lanes[i], lanes[j]) for i, j in zip(*np.nonzero(links)))
    isolated = [lane for lane, n in zip(lanes, links.sum(0) + links.sum(1)) if n == 0]
    print(f"▶ 차로 {len(lanes)}개, 링크 {int(links.sum())}개 (자기 자신 연결 제외)")
    print(f"▶ 약한 연결 요소 {nx.number_weakly_connected_components(G)}개, 고립 차로 {len(isolated)}개 {isolated[:5]}")

    pos = lane_positions(args.net)
    print(f"▶ lanes.txt에만 있는 차로 {len(set(lanes) - set(pos))}개, 도로망에만 있는 차로 {len(set(pos) - set(lanes))}개")

    # 도로망 좌표에 그리기 (좌표가 없는 차로는 spring layout으로 배치)
    missing = [lane for lane in lanes if lane not in pos]
    if missing:
        pos.update(nx.spring_layout(G.subgraph(missing), seed=42))
    isolated_set = set(isolated)
    colors = ['red' if lane in isolated_set else 'blue' for lane in lanes]
    plt.figure(figsize=(8, 8))
    nx.draw_networkx_nodes(G, pos, nodelist=lanes, node_size=20, node_color=colors)
    nx.draw_networkx_edges(G, pos, arrowstyle='->', arrowsize=4, width=0.5)
    plt.legend(handles=[mpatches.Patch(color='red', label='Isolated'), mpatches.Patch(color='blue', label='Connected')],
               loc='upper right')
    plt.title(f"Lane graph (N={len(lanes)}, links={int(links.sum())})")
    plt.axis('off')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
