import unittest
import asyncio

from tinyllm.tests.base import AsyncioTestCase
from tinyllm.functions.function import Function
from tinyllm.functions.concurrent import Concurrent
from tinyllm.state import States


class SleepOperator(Function):
    def __init__(self, sleep_time: float, **kwargs):
        super().__init__(**kwargs)
        self.sleep_time = sleep_time

    async def run(self, *args, **kwargs):
        self.log("Sleeping for {} seconds".format(self.sleep_time))
        await asyncio.sleep(self.sleep_time)
        self.log("Done sleeping")
        return kwargs


class TestParallelOperator(AsyncioTestCase):
    def setUp(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

    def test_parallel_dag(self):
        op1 = SleepOperator(name="SleeperTest1", sleep_time=2)
        op2 = SleepOperator(name="SleeperTest1", sleep_time=2)
        concurrent_dag = Concurrent(name="TestParallel", children=[op1, op2])
        result1 = self.loop.run_until_complete(concurrent_dag(inputs=[{'time': 2},
                                                                      {'time': 2}]))
        self.assertIsNotNone(result1)
        self.assertEqual(result1, {'output': [{'time': 2}, {'time': 2}]})
        self.assertEqual(concurrent_dag.state, States.COMPLETE)

if __name__ == '__main__':
    unittest.main()
