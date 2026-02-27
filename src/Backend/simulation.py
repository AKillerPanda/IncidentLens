import asyncio
import time
from collections import defaultdict
from typing import Dict, Tuple

from src.Backend.graph_data_wrapper import build_window_data
from src.Backend.wrappers import index_flows_bulk

FlowKey = Tuple[str, str, int, int, int]

class StreamSimulator:
    def __init__(self, packets, rate=200, window_size=5.0):
        self.packets = packets
        self.rate = rate
        self.window_size = window_size
        self.active_flows: Dict[FlowKey, dict] = {}
        self.current_window_start = None

    async def run(self):
        interval = 1.0 / self.rate

        for pkt in self.packets:
            await asyncio.sleep(interval)
            self.process_packet(pkt)

    def process_packet(self, pkt):
        ts = pkt["timestamp"]

        if self.current_window_start is None:
            self.current_window_start = ts

        # Window rollover
        if ts >= self.current_window_start + self.window_size:
            self.close_window()
            self.current_window_start = ts

        key = (
            pkt["src_ip"],
            pkt["dst_ip"],
            pkt["src_port"],
            pkt["dst_port"],
            pkt["protocol"],
        )

        flow = self.active_flows.setdefault(key, {
            "packet_count": 0,
            "total_bytes": 0,
            "timestamps": [],
            "label": pkt["label"],
        })

        flow["packet_count"] += 1
        flow["total_bytes"] += pkt["packet_length"]
        flow["timestamps"].append(ts)

    def close_window(self):
        flows = []

        for key, data in self.active_flows.items():
            src_ip, dst_ip, src_port, dst_port, protocol = key

            flows.append({
                "src_ip": src_ip,
                "dst_ip": dst_ip,
                "src_port": src_port,
                "dst_port": dst_port,
                "protocol": protocol,
                "packet_count": data["packet_count"],
                "total_bytes": data["total_bytes"],
                "label": data["label"],
            })

        if flows:
            index_flows_bulk(flows)

        self.active_flows.clear()