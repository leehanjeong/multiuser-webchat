#!/usr/bin/env python3
"""Export Prometheus metrics to CSV for analysis."""

from __future__ import annotations

import argparse
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests


class PrometheusExporter:
    """Exports Prometheus metrics to CSV at regular intervals."""

    def __init__(self, prom_url: str = "http://localhost:9091") -> None:
        self.prom_url = prom_url
        self.running = True
        self.data: list[dict[str, float | str]] = []

        # Metrics to collect
        self.metrics = [
            # User metrics
            "webchat_connected_users",
            # Latency metrics (p95)
            "histogram_quantile(0.95, rate(webchat_message_latency_seconds_bucket[1m]))",
            "histogram_quantile(0.95, rate(webchat_message_e2e_latency_seconds_bucket[1m]))",
            "histogram_quantile(0.95, rate(webchat_eventloop_lag_seconds_bucket[1m]))",
            # Queue/lag metrics
            "webchat_redis_stream_lag_messages",
            "webchat_broadcast_queue_depth",
            # Resource metrics
            "webchat_process_memory_rss_bytes",
            # Throughput metrics
            "rate(webchat_messages_total[1m])",
            "rate(webchat_errors_total[1m])",
            # Redis latency (p95)
            "histogram_quantile(0.95, rate(webchat_redis_latency_seconds_bucket[1m]))",
        ]

        # Metric name mapping for clean column names
        self.column_names = {
            "webchat_connected_users": "connected_users",
            "histogram_quantile(0.95, rate(webchat_message_latency_seconds_bucket[1m]))": "message_latency_p95",
            "histogram_quantile(0.95, rate(webchat_message_e2e_latency_seconds_bucket[1m]))": "e2e_latency_p95",
            "histogram_quantile(0.95, rate(webchat_eventloop_lag_seconds_bucket[1m]))": "eventloop_lag_p95",
            "webchat_redis_stream_lag_messages": "redis_stream_lag_messages",
            "webchat_broadcast_queue_depth": "broadcast_queue_depth",
            "webchat_process_memory_rss_bytes": "memory_rss_bytes",
            "rate(webchat_messages_total[1m])": "message_rate",
            "rate(webchat_errors_total[1m])": "error_rate",
            "histogram_quantile(0.95, rate(webchat_redis_latency_seconds_bucket[1m]))": "redis_latency_p95",
        }

    def query_instant(self, metric: str) -> list[dict]:
        """Query current value of a metric from Prometheus."""
        try:
            resp = requests.get(
                f"{self.prom_url}/api/v1/query",
                params={"query": metric},
                timeout=5,
            )
            resp.raise_for_status()
            result = resp.json()

            if result["status"] == "success":
                return result["data"]["result"]
            else:
                print(f"Error querying {metric}: {result}", file=sys.stderr)
                return []
        except Exception as e:
            print(f"Exception querying {metric}: {e}", file=sys.stderr)
            return []

    def collect_metrics(self) -> dict[str, float | str]:
        """Collect all metrics at current timestamp."""
        row: dict[str, float | str] = {"timestamp": datetime.now().isoformat()}

        for metric in self.metrics:
            clean_name = self.column_names.get(metric, metric)

            try:
                result = self.query_instant(metric)

                if result:
                    # Sum across all instances (for multiprocess metrics)
                    total = sum(float(r["value"][1]) for r in result)
                    row[clean_name] = total
                else:
                    row[clean_name] = 0.0

            except Exception as e:
                print(f"Error processing {metric}: {e}", file=sys.stderr)
                row[clean_name] = 0.0

        return row

    def export_loop(self, interval: int = 15) -> None:
        """Collect metrics every interval seconds until stopped."""
        print(f"Starting metric collection (interval={interval}s)")
        print(f"Prometheus URL: {self.prom_url}")
        print(f"Collecting {len(self.metrics)} metrics")
        print("Press Ctrl+C to stop and save...\n")

        while self.running:
            row = self.collect_metrics()
            self.data.append(row)

            # Print progress
            print(
                f"[{row['timestamp']}] Collected {len(row)-1} metrics "
                f"(users={row.get('connected_users', 0):.0f}, "
                f"latency={row.get('message_latency_p95', 0)*1000:.1f}ms)"
            )

            time.sleep(interval)

    def save(self, output_file: str | Path) -> None:
        """Save collected data to CSV."""
        if not self.data:
            print("No data collected", file=sys.stderr)
            return

        df = pd.DataFrame(self.data)
        df.to_csv(output_file, index=False)

        print(f"\nExported {len(df)} rows to {output_file}")
        print(f"Columns: {', '.join(df.columns)}")
        print(f"Time range: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export Prometheus metrics to CSV during load tests"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="metrics.csv",
        help="Output CSV file path (default: metrics.csv)",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=15,
        help="Collection interval in seconds (default: 15)",
    )
    parser.add_argument(
        "--prom-url",
        type=str,
        default="http://localhost:9091",
        help="Prometheus URL (default: http://localhost:9091)",
    )

    args = parser.parse_args()

    exporter = PrometheusExporter(prom_url=args.prom_url)

    def signal_handler(sig: int, frame: object) -> None:
        print("\n\nStopping metric collection...")
        exporter.running = False

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        exporter.export_loop(interval=args.interval)
    finally:
        exporter.save(args.output)


if __name__ == "__main__":
    main()
