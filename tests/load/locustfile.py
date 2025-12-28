"""Load testing for multiuser webchat with corrected broadcast handling."""

from __future__ import annotations

import json
import random
import threading
import time
from typing import Any

import websocket
from locust import HttpUser, between, constant, constant_pacing, events, task


class ChatUser(HttpUser):
    """
    Normal chat user with background message receiver.

    Message sending rate: ~0.3 msg/sec per user
    Wait time: 3-4 seconds between messages
    """

    host = "http://localhost:8080"
    wait_time = between(3, 4)  # Normal chatting pace

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.ws = None
        self.user_id = f"user_{random.randint(1000, 9999)}"
        self.received_count = 0
        self.listener_thread = None
        self.running = True

    def on_start(self) -> None:
        """Connect to WebSocket and start background receiver."""
        ws_url = self.host.replace("http://", "ws://") + "/ws"
        start_time = time.time()

        try:
            self.ws = websocket.create_connection(ws_url, timeout=10)
            response_time = int((time.time() - start_time) * 1000)

            events.request.fire(
                request_type="WS",
                name="Connect",
                response_time=response_time,
                response_length=0,
                exception=None,
                context={},
            )

            # Start background receiver thread
            self.listener_thread = threading.Thread(
                target=self._receive_loop, daemon=True
            )
            self.listener_thread.start()

        except Exception as e:
            response_time = int((time.time() - start_time) * 1000)
            events.request.fire(
                request_type="WS",
                name="Connect",
                response_time=response_time,
                response_length=0,
                exception=e,
                context={},
            )

    def _receive_loop(self) -> None:
        """Background thread: continuously receive broadcast messages."""
        while self.running and self.ws:
            try:
                start_time = time.time()
                self.ws.settimeout(1.0)  # 1 sec timeout to check running flag
                message = self.ws.recv()

                if message:
                    self.received_count += 1
                    response_time = int((time.time() - start_time) * 1000)

                    events.request.fire(
                        request_type="WS",
                        name="Receive (broadcast)",
                        response_time=response_time,
                        response_length=len(message),
                        exception=None,
                        context={},
                    )

            except websocket.WebSocketTimeoutException:
                continue  # No message, continue listening
            except Exception:
                break  # Connection closed or error

    def on_stop(self) -> None:
        """Stop receiver thread and close WebSocket connection."""
        self.running = False
        if self.listener_thread:
            self.listener_thread.join(timeout=2)
        if self.ws:
            try:
                self.ws.close()
            except Exception:
                pass

    @task
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
            payload = json.dumps(message)
            self.ws.send(payload)
            response_time = int((time.time() - start_time) * 1000)

            events.request.fire(
                request_type="WS",
                name="Send message",
                response_time=response_time,
                response_length=len(payload),
                exception=None,
                context={},
            )
        except Exception as e:
            response_time = int((time.time() - start_time) * 1000)
            events.request.fire(
                request_type="WS",
                name="Send message",
                response_time=response_time,
                response_length=0,
                exception=e,
                context={},
            )


class ActiveUser(ChatUser):
    """
    High activity user (breaking news, hot discussion).

    Message sending rate: ~2.0 msg/sec per user
    Wait time: 0.4-0.6 seconds between messages
    """

    wait_time = between(0.4, 0.6)  # Much faster!


class ModerateUser(ChatUser):
    """
    Moderately active user.

    Message sending rate: ~0.7 msg/sec per user
    Wait time: 1.2-1.6 seconds between messages
    """

    wait_time = between(1.2, 1.6)


class SlowChatUser(ChatUser):
    """
    Slow client simulating poor network or CPU.

    Message sending rate: ~0.15 msg/sec per user
    Receive processing: 0.5-2 second delay
    """

    wait_time = between(6, 8)  # Very slow sending

    def _receive_loop(self) -> None:
        """Slow receiver with processing delay."""
        while self.running and self.ws:
            try:
                start_time = time.time()
                self.ws.settimeout(1.0)
                message = self.ws.recv()

                if message:
                    # Simulate slow processing (poor CPU/network)
                    time.sleep(random.uniform(0.5, 2.0))

                    self.received_count += 1
                    response_time = int((time.time() - start_time) * 1000)

                    events.request.fire(
                        request_type="WS",
                        name="Receive (slow client)",
                        response_time=response_time,
                        response_length=len(message),
                        exception=None,
                        context={},
                    )

            except websocket.WebSocketTimeoutException:
                continue
            except Exception:
                break


class MachineGunUser(ChatUser):
    """
    Phase 2D: Extreme high-frequency user.

    Message sending rate: ~5.0 msg/sec per user
    Wait time: 0.18-0.22 seconds between messages
    """

    wait_time = between(0.18, 0.22)  # Very fast!


class SuperMachineGunUser(ChatUser):
    """
    Phase 2 Level 4: Ultra extreme high-frequency user.

    Message sending rate: ~15.0 msg/sec per user
    Wait time: 0.06-0.07 seconds between messages
    """

    wait_time = between(0.06, 0.07)  # Extremely fast!


class HeavyLifterUser(ChatUser):
    """
    Phase 3: Large message sender.

    Message size controlled by MESSAGE_SIZE environment variable (bytes).
    Default: 1024 bytes (1 KB)
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        import os
        self.message_size = int(os.getenv("MESSAGE_SIZE", "1024"))

    @task
    def send_message(self) -> None:
        """Send a large message via WebSocket."""
        if not self.ws:
            return

        # Create large payload
        message = {
            "text": "X" * self.message_size,
            "type": "message",
            "ts": int(time.time() * 1000),
        }

        start_time = time.time()
        try:
            payload = json.dumps(message)
            self.ws.send(payload)
            response_time = int((time.time() - start_time) * 1000)

            events.request.fire(
                request_type="WS",
                name=f"Send large message ({self.message_size} bytes)",
                response_time=response_time,
                response_length=len(payload),
                exception=None,
                context={},
            )
        except Exception as e:
            response_time = int((time.time() - start_time) * 1000)
            events.request.fire(
                request_type="WS",
                name=f"Send large message ({self.message_size} bytes)",
                response_time=response_time,
                response_length=0,
                exception=e,
                context={},
            )


class BadActor(ChatUser):
    """
    Phase 6A: One malicious user sending large messages at high frequency.

    Message sending rate: ~5.0 msg/sec
    Message size: 100 KB (configurable via BAD_ACTOR_SIZE env var)
    """

    wait_time = between(0.18, 0.22)  # 5 msg/sec

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        import os
        self.message_size = int(os.getenv("BAD_ACTOR_SIZE", str(100 * 1024)))  # 100 KB

    @task
    def send_message(self) -> None:
        """Send a very large message at high frequency."""
        if not self.ws:
            return

        message = {
            "text": "MALICIOUS_" + ("X" * self.message_size),
            "type": "message",
            "ts": int(time.time() * 1000),
        }

        start_time = time.time()
        try:
            payload = json.dumps(message)
            self.ws.send(payload)
            response_time = int((time.time() - start_time) * 1000)

            events.request.fire(
                request_type="WS",
                name="Bad Actor (large+fast)",
                response_time=response_time,
                response_length=len(payload),
                exception=None,
                context={},
            )
        except Exception as e:
            response_time = int((time.time() - start_time) * 1000)
            events.request.fire(
                request_type="WS",
                name="Bad Actor (large+fast)",
                response_time=response_time,
                response_length=0,
                exception=e,
                context={},
            )


class ChurnUser(ChatUser):
    """
    Phase 6B: High churn user - connects and disconnects rapidly.

    Lifetime: 5 seconds
    Simulates mobile users with poor connectivity.
    """

    wait_time = constant(0)  # No waiting between tasks

    def on_start(self) -> None:
        """Connect, wait 5 seconds, then disconnect."""
        super().on_start()

        # Stay connected for 5 seconds
        time.sleep(5)

        # Trigger disconnect
        self.running = False
