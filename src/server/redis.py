from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections.abc import Awaitable, Callable, Generator
from typing import Any

import redis.asyncio as redis
from aiohttp import web

from server.metrics import ERRORS_TOTAL, REDIS_STREAM_LAG_MESSAGES, track_redis_operation
from server.models import ChatMessage, json_dumps, json_loads

logger = logging.getLogger(__name__)


MessageHandler = Callable[[ChatMessage], Awaitable[None]]


SECONDS_IN_MINUTE = 60
MS_IN_SECOND = 1_000


class RedisManager:
    STREAM_KEY = "chat:messages"
    MAX_STREAM_LENGTH = 10_000

    def __init__(self, redis_url: str) -> None:
        self.redis_url = redis_url
        self.client: redis.Redis | None = None
        self._listener_task: asyncio.Task[None] | None = None
        self._message_handler: MessageHandler | None = None
        self._last_id: str = "$"  # Start reading from new messages

    async def connect(self) -> None:
        if self.client:
            # TODO: Change to a warning and just return instead?
            raise RuntimeError("Attempting to connect to Redis client twice!")

        with track_redis_operation("connect"):
            self.client = redis.Redis.from_url(  # pyright: ignore[reportUnknownMemberType]
                url=self.redis_url,
                encoding="utf-8",
                decode_responses=True,
            )

    async def disconnect(self) -> None:
        if self._listener_task and not self._listener_task.done():
            self._listener_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._listener_task

        if self.client:
            with track_redis_operation("disconnect"):
                await self.client.aclose()

    def set_message_handler(self, handler: MessageHandler) -> None:
        self._message_handler = handler

    async def fetch_history(self, minutes: int = 30) -> list[ChatMessage]:
        if self.client is None:
            raise RuntimeError("Redis client not connected!")

        current_time_ms = int(time.time() * MS_IN_SECOND)
        start_time_ms = current_time_ms - (minutes * SECONDS_IN_MINUTE * MS_IN_SECOND)

        # XRANGE with timestamp-based IDs: format is "timestamp-sequence"
        # Using start_time_ms-0 to get all messages from that timestamp onward
        with track_redis_operation("xrange"):
            response = await self.client.xrange(
                name=self.STREAM_KEY,
                min=f"{start_time_ms}-0",
                max="+",  # Current time
            )

        messages = [
            message for _, message in self.extract_messages_from_response(response)
        ]

        return messages

    async def publish_message(self, message: ChatMessage) -> None:
        if self.client is None:
            raise RuntimeError("Redis client not connected!")

        payload = json_dumps(message)
        with track_redis_operation("xadd"):
            await self.client.xadd(
                name=self.STREAM_KEY,
                fields={"data": payload},
                maxlen=self.MAX_STREAM_LENGTH,
                approximate=True,  # More efficient auto-trimming
            )

    async def start_listen(self) -> None:
        if not self.client:
            raise RuntimeError("Redis client not connected!")

        self._listener_task = asyncio.create_task(self._listen_loop())

    async def _listen_loop(self) -> None:
        assert self.client

        try:
            while True:
                try:
                    with track_redis_operation("xread"):
                        stream_data = await self.client.xread(
                            streams={self.STREAM_KEY: self._last_id},
                            count=100,
                            block=5 * MS_IN_SECOND,
                        )
                except Exception:
                    ERRORS_TOTAL.labels(type="redis_error").inc()
                    raise

                # Track Redis Stream lag
                try:
                    stream_info = await self.client.xinfo_stream(self.STREAM_KEY)
                    total_length = stream_info.get('length', 0)
                    REDIS_STREAM_LAG_MESSAGES.set(total_length)
                except Exception:
                    # XINFO can fail on empty stream or if stream doesn't exist yet
                    pass

                if not stream_data:
                    # Timeout, no new messages in last 5 seconds
                    continue

                for _, response in stream_data:
                    messages = self.extract_messages_from_response(response)
                    for message_id, message in messages:
                        self._last_id = message_id
                        if self._message_handler:
                            await self._message_handler(message)
        except asyncio.CancelledError as exc:
            logger.info("Client listener is cancelled.")
            raise exc
        except Exception:
            ERRORS_TOTAL.labels(type="redis_error").inc()
            logger.exception("Unknown exception occurred during listening loop.")
            # Re-raise to fail fast rather than silently stopping message processing
            raise

    def extract_messages_from_response(
        self, response: list[tuple[str, dict[str, Any]]]
    ) -> Generator[tuple[str, ChatMessage]]:
        for message_id, fields in response:
            payload = fields.get("data")
            if not payload:
                logger.warning("Message %s has no 'data' field!", message_id)
                continue

            try:
                chat_message = json_loads(payload)
                if isinstance(chat_message, ChatMessage):
                    yield message_id, chat_message
                else:
                    logger.warning(
                        "Message %s has invalid format: %s", message_id, payload
                    )
            except ValueError:
                logger.exception(
                    "Failed to parse message with id %s and payload %s!",
                    message_id,
                    payload,
                    exc_info=True,
                )
            except Exception:
                logger.warning(
                    "Unknown exception occured while receiving message %s",
                    message_id,
                    exc_info=True,
                )


def install_redis_manager(app: web.Application, redis_url: str) -> RedisManager:
    redis_manager = RedisManager(redis_url)
    app["redis_manager"] = redis_manager
    return redis_manager
