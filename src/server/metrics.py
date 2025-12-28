import os
import time
from collections.abc import Generator
from contextlib import contextmanager, suppress
from pathlib import Path


def setup_multiprocess_dir() -> None:
    """Clean up stale metric files from the multiprocess directory.

    This MUST be called BEFORE importing prometheus_client metrics.
    Note: We only clean files inside the directory, not the directory itself,
    because it may be a mounted tmpfs filesystem.
    """
    prometheus_dir = os.environ.get("PROMETHEUS_MULTIPROC_DIR")
    if not prometheus_dir:
        return

    multiproc_path = Path(prometheus_dir)
    if multiproc_path.exists():
        # Clean files inside the directory, not the directory itself (it might be a mount point)
        for file in multiproc_path.iterdir():
            with suppress(OSError):
                file.unlink()
    else:
        multiproc_path.mkdir(parents=True, exist_ok=True)


# Clean up stale files BEFORE importing prometheus_client
# This ensures files from dead workers don't corrupt metrics
setup_multiprocess_dir()

# ruff: noqa: E402 - Must import after setup_multiprocess_dir()
from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
    multiprocess,
)


def get_metrics_output() -> bytes:
    prometheus_dir = os.environ.get("PROMETHEUS_MULTIPROC_DIR")
    if prometheus_dir:
        registry = CollectorRegistry()
        multiprocess.MultiProcessCollector(registry)  # type: ignore[no-untyped-call]
        return generate_latest(registry)
    else:
        from prometheus_client import REGISTRY

        return generate_latest(REGISTRY)


CONNECTED_USERS = Gauge(
    "webchat_connected_users",
    "Number of currently connected WebSocket users",
    multiprocess_mode="livesum",
)

MESSAGES_TOTAL = Counter(
    "webchat_messages_total",
    "Total number of messages processed",
)

MESSAGE_LATENCY = Histogram(
    "webchat_message_latency_seconds",
    "Time to process a message",
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
)

CONNECTIONS_TOTAL = Counter(
    "webchat_connections_total",
    "Total number of WebSocket connection attempts",
    labelnames=["status"],
)

DISCONNECTIONS_TOTAL = Counter(
    "webchat_disconnections_total",
    "Total number of WebSocket disconnections",
    labelnames=["reason"],
)

REDIS_OPERATIONS_TOTAL = Counter(
    "webchat_redis_operations_total",
    "Total number of Redis operations",
    labelnames=["operation", "status"],
)

REDIS_LATENCY = Histogram(
    "webchat_redis_latency_seconds",
    "Redis operation latency",
    labelnames=["operation"],
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
)

ERRORS_TOTAL = Counter(
    "webchat_errors_total",
    "Total number of errors",
    labelnames=["type"],
)

# ===== New Metrics for Failure Detection =====

# 1. Event Loop Health
EVENTLOOP_LAG_SECONDS = Histogram(
    "webchat_eventloop_lag_seconds",
    "Event loop responsiveness (time to execute zero-delay callback)",
    buckets=(0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0),
)

# 2. Redis Stream Backlog
REDIS_STREAM_LAG_MESSAGES = Gauge(
    "webchat_redis_stream_lag_messages",
    "Unprocessed messages in Redis Stream",
    multiprocess_mode="livesum",
)

# 3. Broadcast Queue Depth
BROADCAST_QUEUE_DEPTH = Gauge(
    "webchat_broadcast_queue_depth",
    "Messages waiting to be broadcast to clients",
    multiprocess_mode="livesum",
)

# 4. Process Memory
PROCESS_MEMORY_RSS_BYTES = Gauge(
    "webchat_process_memory_rss_bytes",
    "Process resident memory in bytes",
    multiprocess_mode="livesum",
)

# 5. End-to-End Message Latency
MESSAGE_E2E_LATENCY_SECONDS = Histogram(
    "webchat_message_e2e_latency_seconds",
    "Time from message creation to broadcast completion",
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0),
)


@contextmanager
def track_redis_operation(operation: str) -> Generator[None]:
    start_time = time.time()
    try:
        yield
        REDIS_OPERATIONS_TOTAL.labels(operation=operation, status="success").inc()
    except Exception as e:
        REDIS_OPERATIONS_TOTAL.labels(operation=operation, status="error").inc()
        raise e
    finally:
        elapsed = time.time() - start_time
        REDIS_LATENCY.labels(operation=operation).observe(elapsed)


@contextmanager
def track_message_processing() -> Generator[None]:
    start_time = time.time()
    try:
        yield
        MESSAGES_TOTAL.inc()
    finally:
        elapsed = time.time() - start_time
        MESSAGE_LATENCY.observe(elapsed)
