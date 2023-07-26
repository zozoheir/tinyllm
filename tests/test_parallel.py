import unittest
import asyncio

from tinyllm.functions.chain import Chain
from tinyllm.functions.function import Function
from tinyllm.functions.parallel import Concurrent


class SleepOperator(Function):
    def __init__(self, sleep_time: float, **kwargs):
        super().__init__(**kwargs)
        self.sleep_time = sleep_time

    async def run(self, *args, **kwargs):
        self.log("Sleeping for {} seconds".format(self.sleep_time))
        await asyncio.sleep(self.sleep_time)
        self.log("Done sleeping")
        return kwargs


class TestParallelOperator(unittest.TestCase):
    def setUp(self):
        self.loop = asyncio.get_event_loop()

    def test_parallel_dag(self):
        op1 = SleepOperator(name="SleeperTest1", sleep_time=2)
        op2 = SleepOperator(name="SleeperTest1", sleep_time=2)
        concurrent_dag = Concurrent(name="TestParallel", children=[op1, op2])
        result1 = self.loop.run_until_complete(concurrent_dag(inputs=[{'time': 2},
                                                                   {'time': 2}]))
        self.assertIsNotNone(result1)


if __name__ == '__main__':
    unittest.main()
