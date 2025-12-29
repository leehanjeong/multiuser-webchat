import argparse
import asyncio
import logging
import multiprocessing
import os
import signal
import sys
import time
from pathlib import Path
from types import FrameType

from aiohttp import web

from server.metrics import (
    EVENTLOOP_LAG_SECONDS,
    PROCESS_MEMORY_RSS_BYTES,
    get_metrics_output,
)
from server.models import json_dumps
from server.redis import RedisManager, install_redis_manager
from server.ws import WSMessageRouter, install_ws_router

logger = logging.getLogger(__name__)


STATIC_RESOURCES_DIR = Path(__file__).resolve().parent.parent / "static"
MINUTES_IN_HOUR = 60
HOUR_IN_DAY = 24

background_tasks = set()


async def _monitor_eventloop_lag() -> None:
    """Measure event loop responsiveness every 1 second."""
    while True:
        start = time.perf_counter()
        await asyncio.sleep(0)  # Zero-delay callback to measure lag
        lag = time.perf_counter() - start
        EVENTLOOP_LAG_SECONDS.observe(lag)
        await asyncio.sleep(1)


async def _monitor_memory() -> None:
    """Track process memory usage every 15 seconds."""
    try:
        import psutil  # type: ignore
    except ImportError:
        logger.warning("psutil not installed, skipping memory monitoring")
        return

    process = psutil.Process()
    while True:
        try:
            memory_info = process.memory_info()
            PROCESS_MEMORY_RSS_BYTES.set(memory_info.rss)
        except Exception:
            logger.exception("Error collecting memory metrics")
        await asyncio.sleep(15)


async def on_startup(app: web.Application) -> None:
    redis_manager: RedisManager = app["redis_manager"]
    logger.info("Connecting to Redis...")
    await redis_manager.connect()
    logger.info("Redis connected! Starting stream listener...")
    await redis_manager.start_listen()
    logger.info("The server is now ready to listen to Redis stream messages!")

    # Start monitoring tasks
    task1 = asyncio.create_task(_monitor_eventloop_lag())
    background_tasks.add(task1)
    task1.add_done_callback(background_tasks.discard)  # 작업 완료 시 set에서 제거

    task2 = asyncio.create_task(_monitor_memory())
    background_tasks.add(task2)
    task2.add_done_callback(background_tasks.discard)
    logger.info("Started event loop and memory monitoring tasks")


async def on_cleanup(app: web.Application) -> None:
    ws_router: WSMessageRouter = app["ws_router"]
    logger.info("Closing all open WebSocket connections...")
    await ws_router.close_all_connections()
    logger.info("Successfully closed all WebSocket connections!")

    redis_manager: RedisManager = app["redis_manager"]
    logger.info("Disconnecting to Redis...")
    await redis_manager.disconnect()
    logger.info("Successfully disconnected to Redis!")


async def healthz(_: web.Request) -> web.Response:
    return web.json_response({"status": "ok"})


async def metrics(_: web.Request) -> web.Response:
    metrics_output = get_metrics_output()
    return web.Response(
        body=metrics_output,
        content_type="text/plain; version=0.0.4",
        charset="utf-8",
    )


async def index(_: web.Request) -> web.FileResponse:
    return web.FileResponse(STATIC_RESOURCES_DIR / "index.html")


async def get_messages(req: web.Request) -> web.Response:
    minutes_str = req.query.get("minutes", "30")
    if not minutes_str.isdigit():
        raise web.HTTPBadRequest(reason="minutes is not a valid integer!")
    minutes = int(minutes_str)
    if minutes < 1:
        raise web.HTTPBadRequest(reason="minutes must be a positive number!")
    if minutes > MINUTES_IN_HOUR * HOUR_IN_DAY:
        raise web.HTTPBadRequest(reason="minutes cannot be more than 24 hours!")

    redis_manager: RedisManager = req.app["redis_manager"]
    try:
        messages = await redis_manager.fetch_history(minutes=minutes)
        return web.json_response({"messages": messages}, dumps=json_dumps)
    except Exception as exc:
        logger.exception("Failed to fetch message history")
        return web.json_response({"error": str(exc)}, status=500)


def create_app(redis_url: str) -> web.Application:
    app = web.Application()
    app.add_routes(
        [
            web.get("/", index),
            web.get("/healthz", healthz),
            web.get("/metrics", metrics),
            web.get("/messages", get_messages),
            web.static("/static", STATIC_RESOURCES_DIR),
        ]
    )

    redis_manager = install_redis_manager(app, redis_url)
    install_ws_router(app, redis_manager)

    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)

    return app


def run_worker(worker_id: int, host: str, port: int, redis_url: str) -> None:
    setup_logging()

    logger.info(f"[Worker {worker_id}] Starting on {host}:{port} (PID: {os.getpid()})")

    app = create_app(redis_url)

    web.run_app(
        app,
        host=host,
        port=port,
        reuse_port=True,
        print=lambda *args: None,  # Suppress aiohttp's startup message per worker
    )


def shutdown_workers(
    processes: list[multiprocessing.Process], timeout: int = 5
) -> None:
    logger.info("Shutting down %d workers...", len(processes))

    for process in processes:
        if process.is_alive():
            process.terminate()

    for process in processes:
        process.join(timeout=timeout)

        if process.is_alive():
            logger.warning(
                "Worker %s didn't stop after %d seconds, killing...",
                timeout,
                process.name,
            )
            process.kill()
            process.join()

    logger.info("All %d workers stopped", len(processes))


def register_signal_handlers(processes: list[multiprocessing.Process]) -> None:
    def signal_handler(signum: int, _: FrameType | None) -> None:
        logger.info("Received signal %d", signum)
        shutdown_workers(processes)
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="A multiuser webchat application")

    parser.add_argument(
        "--host",
        type=str,
        default=os.getenv("HOST", "localhost"),
        help="Host to bind to (default: localhost, env: HOST)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("PORT", "8080")),
        help="Port to bind to (default: 8080, env: PORT)",
    )
    parser.add_argument(
        "--redis-url",
        type=str,
        default=os.getenv("REDIS_URL", "redis://localhost:6379"),
        help="Redis connection URL (default: redis://localhost:6379, env: REDIS_URL)",
    )

    def positive_integer(value: str) -> int:
        try:
            ivalue = int(value)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"{value} is not a valid integer") from exc
        if ivalue < 1:
            raise argparse.ArgumentTypeError(f"{value} must be at least 1")
        return ivalue

    parser.add_argument(
        "--workers",
        type=positive_integer,
        default=positive_integer(os.getenv("WORKERS", "1")),
        help="Number of workers (must be at least 1, default: 1, env: WORKERS)",
    )

    return parser.parse_args()


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[{levelname:<8}] {asctime} [PID:{process}] ({name}.{funcName}) {message}",
        style="{",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main() -> None:
    args = parse_args()
    host: str = args.host
    port: int = args.port
    redis_url: str = args.redis_url
    workers: int = args.workers

    if workers == 1:
        logger.info("Starting server with address %s:%s...", host, port)
        web.run_app(create_app(redis_url), host=host, port=port)
        return

    processes: list[multiprocessing.Process] = []
    for worker_id in range(workers):
        process = multiprocessing.Process(
            target=run_worker,
            args=(worker_id, host, port, redis_url),
            name=f"worker-{worker_id}",
        )
        process.start()
        logger.info("Started worker %d (PID: %d)", worker_id, process.pid)

        processes.append(process)

    register_signal_handlers(processes)

    try:
        for process in processes:
            process.join()

        shutdown_workers(processes)
    except KeyboardInterrupt:
        # Signal handler is already called at this point.
        pass


if __name__ == "__main__":
    setup_logging()
    main()
