from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from enum import Enum, auto

from aiohttp import WSCloseCode, WSMessage, WSMsgType, web

from server.metrics import (
    BROADCAST_QUEUE_DEPTH,
    CONNECTED_USERS,
    CONNECTIONS_TOTAL,
    DISCONNECTIONS_TOTAL,
    ERRORS_TOTAL,
    MESSAGE_E2E_LATENCY_SECONDS,
    track_message_processing,
)
from server.models import ChatMessage, json_dumps, json_loads
from server.redis import RedisManager

logger = logging.getLogger(__name__)


class PeerStatus(Enum):
    OK = auto()
    CLOSED = auto()
    TIMEOUT = auto()
    INTERNAL_ERROR = auto()


WS_CLOSE_TIMEOUT = 2.0
SEND_TIMEOUT = 0.25
WS_HEARTBEAT_INTERVAL = 25

# WebSocket close codes
WS_CLOSE_NORMAL = 1000
WS_CLOSE_GOING_AWAY = 1001


class WSMessageRouter:
    def __init__(self, redis_manager: RedisManager) -> None:
        self.clients: set[web.WebSocketResponse] = set()
        self.redis = redis_manager
        self._active_broadcasts = 0

        self.redis.set_message_handler(self._broadcast_to_local_peers)

    async def handler(self, req: web.Request) -> web.StreamResponse:
        async with self._initialize_ws(req) as ws:
            async for message in ws:
                match message.type:
                    case WSMsgType.TEXT:
                        logger.info("Received message %s of type TEXT.", message)
                        await self._handle_text(message)
                    case WSMsgType.ERROR:
                        ERRORS_TOTAL.labels(type="websocket_error").inc()
                        logger.error("Received an Error message %s!", message)
                        break
                    case _ as msg_type:
                        # TODO: Handle more WSMsgType.
                        logger.warning(
                            "Unknown message type %s! message=%s", msg_type, message
                        )
                        break
        return ws

    @asynccontextmanager
    async def _initialize_ws(
        self, req: web.Request
    ) -> AsyncGenerator[web.WebSocketResponse]:
        ws = web.WebSocketResponse(heartbeat=WS_HEARTBEAT_INTERVAL)
        logger.info("Connecting to WebSocket at address %s...", req.url)

        try:
            await ws.prepare(req)
        except Exception:
            CONNECTIONS_TOTAL.labels(status="upgrade_failed").inc()
            raise

        logger.info("Connection established to %s!", req.url)
        CONNECTIONS_TOTAL.labels(status="success").inc()
        CONNECTED_USERS.inc()
        self.clients.add(ws)

        disconnect_reason = "normal"
        try:
            yield ws
        except TimeoutError:
            disconnect_reason = "timeout"
            raise
        except Exception:
            disconnect_reason = "error"
            raise
        finally:
            self.clients.discard(ws)
            CONNECTED_USERS.dec()

            if ws.close_code == WSCloseCode.GOING_AWAY:
                disconnect_reason = "going_away"
            elif (
                ws.close_code
                and ws.close_code >= WS_CLOSE_NORMAL
                and disconnect_reason == "normal"
            ):
                disconnect_reason = (
                    "server_shutdown"
                    if ws.close_code == WS_CLOSE_GOING_AWAY
                    else "normal"
                )

            DISCONNECTIONS_TOTAL.labels(reason=disconnect_reason).inc()

            logger.info("Disconnecting from WebSocket...")
            await ws.close()
            logger.info("Successfully disconnected.")

    async def close_all_connections(self) -> None:
        if not self.clients:
            logger.info("No active WebSocket connections to close")
            return

        logger.info("Closing %d active WebSocket connection(s)...", len(self.clients))

        try:
            clients_snapshot = tuple(self.clients)
            closing_open_sockets = (
                ws.close(code=WSCloseCode.GOING_AWAY, message=b"Server shutting down")
                for ws in clients_snapshot
                if not ws.closed
            )
            await asyncio.wait_for(
                asyncio.gather(*closing_open_sockets, return_exceptions=True),
                timeout=WS_CLOSE_TIMEOUT,
            )
        except TimeoutError:
            # Best-effort socket shutdown. Process is terminating anyway.
            logger.warning("Timeout while closing WebSocket connections")

        self.clients.clear()

    async def _handle_text(self, message: WSMessage) -> None:
        data = message.data

        try:
            obj = json_loads(data)
        except ValueError:
            ERRORS_TOTAL.labels(type="parse_error").inc()
            logger.warning("Failed to parse message %s", data, exc_info=True)
            return
        except Exception:
            ERRORS_TOTAL.labels(type="parse_error").inc()
            logger.exception("Unexpected error while parsing message %s")
            return

        if not isinstance(obj, ChatMessage):
            ERRORS_TOTAL.labels(type="invalid_message").inc()
            logger.warning("Received invalid message format: %s", data)
            return

        with track_message_processing():
            await self.redis.publish_message(obj)

    async def _broadcast_to_local_peers(self, message: ChatMessage) -> None:
        # Track broadcast queue depth
        self._active_broadcasts += 1
        BROADCAST_QUEUE_DEPTH.set(self._active_broadcasts)

        try:
            # Calculate end-to-end latency
            message_created_ms = message.ts
            broadcast_start_ms = int(time.time() * 1000)
            e2e_latency_s = (broadcast_start_ms - message_created_ms) / 1000.0
            MESSAGE_E2E_LATENCY_SECONDS.observe(e2e_latency_s)

            payload = json_dumps(message)

            # Snapshotting the clients set is necessary here, as during await a
            # client can disconnect, causing a mutation of the clients set, which
            # will cause the iteration to fail.
            clients_snapshot = tuple(self.clients)

            broadcast_results = await asyncio.gather(
                *(self._send_to_peer(peer, payload) for peer in clients_snapshot)
            )

            for peer, result in zip(clients_snapshot, broadcast_results, strict=True):
                if result != PeerStatus.OK:
                    self.clients.discard(peer)
        finally:
            # Decrement broadcast counter
            self._active_broadcasts -= 1
            BROADCAST_QUEUE_DEPTH.set(self._active_broadcasts)

    async def _send_to_peer(
        self, peer: web.WebSocketResponse, payload: str
    ) -> PeerStatus:
        if peer.closed:
            logger.info("Connection to %s is closed.", peer)
            return PeerStatus.CLOSED

        try:
            await asyncio.wait_for(peer.send_str(payload), timeout=SEND_TIMEOUT)
        except TimeoutError:
            logger.warning(
                "Connection to %s timed out after %s seconds while sending message %s.",
                peer,
                SEND_TIMEOUT,
                payload,
            )
            # add_done_callback prevents 'Task was destroyed but pending' warnings
            asyncio.create_task(
                peer.close(
                    code=WSCloseCode.GOING_AWAY,
                    message=b"Send timeout",
                )
            ).add_done_callback(lambda _: _)
            return PeerStatus.TIMEOUT
        except Exception:
            # TODO: Hard-exit for unexpected error?
            logger.exception(
                "Unknown internal error for %s while sending message %s!",
                peer,
                payload,
            )
            # add_done_callback prevents 'Task was destroyed but pending' warnings
            asyncio.create_task(
                peer.close(
                    code=WSCloseCode.INTERNAL_ERROR,
                    message=b"Unknown internal error",
                )
            ).add_done_callback(lambda _: _)
            return PeerStatus.INTERNAL_ERROR

        return PeerStatus.OK


def install_ws_router(app: web.Application, redis_manager: RedisManager) -> None:
    router = WSMessageRouter(redis_manager)
    app["ws_router"] = router
    app.router.add_get("/ws", router.handler)
