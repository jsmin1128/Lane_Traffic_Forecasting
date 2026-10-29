"""
SUMO 시뮬레이션 입력 파일을 날짜별로 만듭니다 (processed/sumo/<날짜>/). 학습에는 쓰지 않습니다.

  demand.rou.xml      회전교통량(turn.parquet)으로 만든 차량 흐름(flow)
  detectors.add.xml   모든 차로의 E2 검지기
  edgeGroups.add.xml  KPI 측정용 edge 그룹
  tls_plan.tll.xml    신호(signal_xml.parquet)의 현시별 평균 시간
  bucheon.net.xml, exp.rou.xml  원본 도로망·차종 정의 복사본
  bucheon.sumocfg     위 파일을 묶은 SUMO 설정
"""

import shutil
from datetime import datetime
from pathlib import Path
from string import Template
from zoneinfo import ZoneInfo

import pyarrow.parquet as pq
import sumolib

NET_FILE  = "bucheon.net.xml"
EXP_ROUTE = "exp.rou.xml"

# map raw veh_type codes to defined vType ids in exp.rou.xml
VTYPE_MAP = {
    'p': '1',          # passenger
    'b': '20',         # bus
    't': '10',         # truck
    'st': '10',        # small truck
    'lt': '10',        # large truck
    's': 'emergency',  # special/emergency vehicles
    'm': '40'          # motorcycle
}

# template for bucheon.sumocfg
CFG_TMPL = Template(
    '<configuration>\n'
    '  <input>\n'
    '    <net-file value="$net"/>\n'
    '    <route-files value="$exp,demand.rou.xml"/>\n'
    '    <additional-files value="detectors.add.xml,edgeGroups.add.xml,tls_plan.tll.xml"/>\n'
    '  </input>\n'
    '  <time begin="0" end="86400"/>\n'
    '</configuration>\n'
)


def build_sumo(raw_dir, out_dir, dates):
    raw_dir, out_dir = Path(raw_dir), Path(out_dir)
    turn_pq = raw_dir / "turn.parquet"
    sig_pq  = raw_dir / "signal_xml.parquet"
    net_src = raw_dir / "net" / NET_FILE
    exp_src = raw_dir / "net" / EXP_ROUTE

    # load net edges and lanes once for detectors
    net_obj = sumolib.net.readNet(str(net_src))
    lane_list = []
    lane_lengths = {}
    for edge in net_obj.getEdges():
        for lane in edge.getLanes():
            lid = lane.getID()
            lane_list.append(lid)
            lane_lengths[lid] = lane.getLength()
    # Build set of valid edge_from → edge_to pairs (direct connections)
    valid_pairs = set()
    for edge in net_obj.getEdges():
        for succ in edge.getOutgoing():
            valid_pairs.add((edge.getID(), succ.getID()))
    edge_ids = [edge.getID() for edge in net_obj.getEdges()]

    for date in map(str, dates):
        # 1) demand.rou.xml
        df_turn = (
            pq.ParquetDataset(turn_pq, filters=[('date', '=', int(date))])
            .read(columns=['dt', 'edge_from', 'edge_to', 'veh_type', 'count', 'begin', 'end'])
            .to_pandas()
        )
        if df_turn.empty:
            print(f'[SUMOPREP] No turn data for date {date}, skipping.')
            continue
        date_dir = out_dir / date
        date_dir.mkdir(parents=True, exist_ok=True)

        # simulation starts at local (KST) midnight of this date
        date_epoch = int(datetime.strptime(date, "%Y%m%d").replace(tzinfo=ZoneInfo('Asia/Seoul')).timestamp())

        # ensure flows are sorted by departure time (stable: same order on every run)
        df_turn['begin'] = df_turn['begin'].astype(float)
        df_turn = df_turn.sort_values(by='begin', kind='stable')
        with open(date_dir / 'demand.rou.xml', 'w', encoding='utf-8') as f:
            f.write('<routes>\n')
            for i, (_, r) in enumerate(df_turn.iterrows()):
                vt = VTYPE_MAP.get(r.veh_type, r.veh_type)
                # convert absolute UNIX time to seconds since simulation start
                beg = int(float(r.begin)) - date_epoch
                end = int(float(r.end)) - date_epoch
                count = int(r['count'])
                if count <= 0:
                    continue
                if beg < 0 or beg > 86400:
                    continue
                # Only write flow if edge_from and edge_to are directly connected
                if (r.edge_from, r.edge_to) in valid_pairs:
                    f.write(
                        f'  <flow id="f_{r.edge_from}_{r.edge_to}_{beg}_{i}" '
                        f'type="{vt}" begin="{beg}" end="{end}" '
                        f'from="{r.edge_from}" to="{r.edge_to}" number="{count}"/>\n'
                    )
            f.write('</routes>\n')

        # 2) detectors.add.xml (E2 detectors on every lane)
        with open(date_dir / 'detectors.add.xml', 'w', encoding='utf-8') as f:
            f.write('<additional>\n')
            for lid in lane_list:
                length = lane_lengths.get(lid, 0)
                f.write(
                    f'  <e2Detector id="det_{lid}" lane="{lid}" pos="0" endPos="{length}" freq="60" '
                    f'file="detectors_output.xml"/>\n'
                )
            f.write('</additional>\n')

        # 3) edgeGroups.add.xml (관심 구간 KPI 그룹: 모든 edge)
        with open(date_dir / 'edgeGroups.add.xml', 'w', encoding='utf-8') as f:
            f.write('<additional>\n')
            f.write(f'  <edgeGroup id="all_edges" edges="{" ".join(edge_ids)}"/>\n')
            f.write('</additional>\n')

        # 4) tls_plan.tll.xml
        df_sig = (
            pq.ParquetDataset(sig_pq, filters=[('date', '=', int(date))])
            .read(columns=['id', 'phase_state', 'phase_duration'])
            .to_pandas()
        )
        avg_dur = df_sig.groupby(['id', 'phase_state'])['phase_duration'].mean()
        with open(date_dir / 'tls_plan.tll.xml', 'w', encoding='utf-8') as f:
            f.write('<additional>\n')
            for (iid, state), dur in avg_dur.items():
                f.write(f'  <phase duration="{int(dur)}" state="{state}"/>\n')
            f.write('</additional>\n')

        # 5) copy net and exp.rou.xml
        shutil.copy(net_src, date_dir / NET_FILE)
        shutil.copy(exp_src, date_dir / EXP_ROUTE)

        # 6) bucheon.sumocfg
        with open(date_dir / 'bucheon.sumocfg', 'w', encoding='utf-8') as f:
            f.write(CFG_TMPL.substitute(net=NET_FILE, exp=EXP_ROUTE))

        print(f'[SUMOPREP] Generated for date {date}')
