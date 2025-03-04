import asyncio
import time


class DynamicThrottle:
    _throttle_startup_count: int = 20
    _throttle_alpha: float = 1.0 / _throttle_startup_count

    def __init__(self, tps: float, num_threads: int = 1) -> None:
        """
        Initialize a DynamicThrottle instance with a target TPS (transactions per second).

        :param tps: Target transactions per second (0 means no throttling)
        :param num_threads: Number of concurrent threads (default: 1)
        """
        if tps == 0:
            # No throttling
            self.target_period: float = 0
        else:
            # Calculate the target period in seconds
            self.target_period: float = num_threads / tps

        self.avg_fn_delay: float = 0.0
        self.n_records: int = 0
        self.last_record_timestamp: float = 0.0
        self.target_tps : float = tps
        self.tps_state : str = None

    @staticmethod
    def ramp(value: float) -> float:
        """
        Ensure non-negative value for pause.

        :param value: The value to validate.
        :return: Non-negative value.
        """
        return max(0.0, value)

    def pause_for_duration(self) -> float:
        """
        Calculate pause duration based on the current record timestamp.

        :return: Pause duration in seconds.
        """

        # Get the current time in seconds
        current_record_timestamp: float = time.time()

        if self.n_records < self._throttle_startup_count:
            # During initial calls
            if self.n_records == 0:
                pause_for = self.target_period
            else:
                alpha = 1.0 / self.n_records
                avg = self.avg_fn_delay
                avg = (1 - alpha) * avg + alpha * (current_record_timestamp - self.last_record_timestamp)
                self.avg_fn_delay = avg
                pause_for = self.target_period - avg
        else:
            # After sufficient records have been logged
            avg = self.avg_fn_delay
            avg = (1 - self._throttle_alpha) * avg + self._throttle_alpha * (current_record_timestamp - self.last_record_timestamp)
            self.avg_fn_delay = avg
            pause_for = self.target_period - avg

        # Ensure non-negative pause
        pause_for = self.ramp(pause_for)

        # Update last record and record count
        self.last_record_timestamp = current_record_timestamp + pause_for
        self.n_records += 1
        return pause_for

    def reset(self) -> None:
        self.tps_state = None

    async def throttle(self) -> None:
        """
        Throttle execution to maintain the target period.
        """
        if self.target_period == 0:
            return

        pause_duration: float = self.pause_for_duration()

        if pause_duration > 0:
            self.tps_state = f'Throttled ({pause_duration} secs)'
            # Sleep for the calculated duration in seconds
            await asyncio.sleep(pause_duration)
