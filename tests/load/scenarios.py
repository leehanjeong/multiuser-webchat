"""Load test scenario shapes for different testing patterns."""

from __future__ import annotations

from locust import LoadTestShape


class BaselineLoad(LoadTestShape):
    """
    Baseline load test: 100 users for 5 minutes.

    Purpose: Establish performance baseline for normal operations.
    Expected: Stable latency, no errors, consistent throughput.
    """

    def tick(self) -> tuple[int, float] | None:
        run_time = self.get_run_time()

        if run_time < 60:
            # Ramp up: 0 -> 100 users in 1 minute
            user_count = int(run_time / 60 * 100)
            return (user_count, 10)
        elif run_time < 360:
            # Sustain: 100 users for 5 minutes
            return (100, 10)
        else:
            # Stop test
            return None


class StressTest(LoadTestShape):
    """
    Stress test: 100 -> 1000 users over 10 minutes.

    Purpose: Find system breaking point and performance degradation patterns.
    Expected: Gradual latency increase, identify capacity limits.
    """

    def tick(self) -> tuple[int, float] | None:
        run_time = self.get_run_time()

        if run_time < 600:
            # Ramp up: 100 -> 1000 users over 10 minutes
            # Add 100 users every minute
            user_count = 100 + int((run_time / 60) * 100)
            user_count = min(user_count, 1000)
            return (user_count, 50)
        elif run_time < 900:
            # Sustain peak: 1000 users for 5 minutes
            return (1000, 50)
        else:
            # Stop test
            return None


class SpikeTest(LoadTestShape):
    """
    Spike test: Sudden traffic spikes.

    Purpose: Test system behavior under sudden load changes.
    Pattern: 100 -> 500 -> 100 -> 500 users
    """

    def tick(self) -> tuple[int, float] | None:
        run_time = self.get_run_time()

        if run_time < 60:
            # Initial: 100 users
            return (100, 10)
        elif run_time < 120:
            # Spike: 500 users
            return (500, 50)
        elif run_time < 240:
            # Drop: 100 users
            return (100, 10)
        elif run_time < 300:
            # Spike again: 500 users
            return (500, 50)
        elif run_time < 420:
            # Sustain: 100 users
            return (100, 10)
        else:
            # Stop test
            return None


class SlowClientTest(LoadTestShape):
    """
    Slow client test: Mix of normal and slow clients.

    Purpose: Test backpressure handling with slow clients.
    Pattern: 90 normal users + 10 slow users
    """

    def tick(self) -> tuple[int, float] | None:
        run_time = self.get_run_time()

        if run_time < 60:
            # Ramp up
            user_count = int(run_time / 60 * 100)
            return (user_count, 10)
        elif run_time < 360:
            # Sustain: 90% normal, 10% slow
            # Note: Actual slow client ratio controlled in locustfile
            return (100, 10)
        else:
            # Stop test
            return None
