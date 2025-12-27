"""Load testing for multiuser webchat with WebSocket support."""

from __future__ import annotations

import json
import random
import time
from typing import Any

from locust import HttpUser, between, events, task
from locust.contrib.fasthttp import FastHttpUser


class ChatUser(FastHttpUser):
    """Simulates a normal chat user with WebSocket connection."""

    host = "http://localhost:8080"
    wait_time = between(1, 5)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.ws = None
        self.user_id = f"user_{random.randint(1000, 9999)}"

    def on_start(self) -> None:
        """Connect to WebSocket when user starts."""
        try:
            self.ws = self.client.ws_connect(
                "/ws",
                timeout=10,
            )
        except Exception as e:
            events.request.fire(
                request_type="WS_CONNECT",
                name="/ws",
                response_time=0,
                response_length=0,
                exception=e,
                context={},
            )

    def on_stop(self) -> None:
        """Close WebSocket when user stops."""
        if self.ws:
            try:
                self.ws.close()
            except Exception:
                pass

    @task(10)
    def send_message(self) -> None:
        """Send a chat message via WebSocket."""
        if not self.ws:
            return

        message = {
            "text": f"Message from {self.user_id} at {time.time()}",
            "type": "message",
            "ts": int(time.time() * 1000),
        }

        start_time = time.time()
        try:
            self.ws.send(json.dumps(message))
            response_time = int((time.time() - start_time) * 1000)

            events.request.fire(
                request_type="WS_SEND",
                name="/ws send_message",
                response_time=response_time,
                response_length=len(json.dumps(message)),
                exception=None,
                context={},
            )
        except Exception as e:
            response_time = int((time.time() - start_time) * 1000)
            events.request.fire(
                request_type="WS_SEND",
                name="/ws send_message",
                response_time=response_time,
                response_length=0,
                exception=e,
                context={},
            )

    @task(2)
    def receive_message(self) -> None:
        """Receive messages from WebSocket."""
        if not self.ws:
            return

        start_time = time.time()
        try:
            # Non-blocking receive with timeout
            result = self.ws.recv()
            if result:
                response_time = int((time.time() - start_time) * 1000)
                events.request.fire(
                    request_type="WS_RECV",
                    name="/ws receive_message",
                    response_time=response_time,
                    response_length=len(result) if isinstance(result, str) else 0,
                    exception=None,
                    context={},
                )
        except Exception as e:
            response_time = int((time.time() - start_time) * 1000)
            events.request.fire(
                request_type="WS_RECV",
                name="/ws receive_message",
                response_time=response_time,
                response_length=0,
                exception=e,
                context={},
            )


class SlowChatUser(ChatUser):
    """Simulates a slow client with poor network or CPU."""

    wait_time = between(3, 8)

    @task(1)
    def receive_message_slowly(self) -> None:
        """Receive messages with artificial delay."""
        if not self.ws:
            return

        start_time = time.time()
        try:
            result = self.ws.recv()
            if result:
                # Simulate slow processing (poor CPU/network)
                time.sleep(random.uniform(1.0, 3.0))

                response_time = int((time.time() - start_time) * 1000)
                events.request.fire(
                    request_type="WS_RECV_SLOW",
                    name="/ws receive_message (slow)",
                    response_time=response_time,
                    response_length=len(result) if isinstance(result, str) else 0,
                    exception=None,
                    context={},
                )
        except Exception as e:
            response_time = int((time.time() - start_time) * 1000)
            events.request.fire(
                request_type="WS_RECV_SLOW",
                name="/ws receive_message (slow)",
                response_time=response_time,
                response_length=0,
                exception=e,
                context={},
            )
