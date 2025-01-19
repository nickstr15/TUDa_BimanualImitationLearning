import time


class RealTimeHandler:
    """
    Handler trying to run loops at a desired frequency.

    Note: This handler guarantees that the loop runs with a maximum frequency of `frequency`,
    but is very likely to run slower than the desired frequency.

    Example usage:

    rt = RealTimeHandler(render_fps)
    rt.reset() <-- call this directly before loop
    for i in range(1000):
        ################################
        # do something that takes time #
        # e.g. rendering               #
        ################################
        rt.sleep() <-- call this at the end of the loop
    """
    def __init__(self, frequency: float = 1.0):
        """
        Initialize the handler.

        Example usage:

        rt = RealTimeHandler(render_fps)
        rt.reset() <-- call this directly before loop
        for i in range(1000):
            ################################
            # do something that takes time #
            # e.g. rendering               #
            ################################
            rt.sleep() <-- call this at the end of the loop

        :param frequency: The desired frequency
        """
        self._frequency = frequency
        self._dt = 1.0 / frequency
        self._start_time = time.time()

    def reset(self) -> None:
        """
        Reset the start time.
        Call this **directly** before the loop
        """
        self._start_time = time.time()

    def sleep(self) -> None:
        """
        Sleep to maintain the desired frequency.
        Call this at the very end of the loop.
        :return:
        """
        elapsed_time = time.time() - self._start_time
        sleep_time = self._dt - elapsed_time

        sleep_time = max(0.0, sleep_time)
        time.sleep(sleep_time)

        # prepare for the next iteration
        self._start_time = time.time()
