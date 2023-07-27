import unittest
import asyncio

from tests.base import AsyncioTestCase
from tinyllm.functions.chain import Chain
from tinyllm.functions.function import Function
from tinyllm.functions.concurrent import Concurrent


class SleepOperator(Function):
    def __init__(self, sleep_time: float, **kwargs):
        super().__init__(**kwargs)
        self.sleep_time = sleep_time

    async def run(self, *args, **kwargs):
        self.log("Sleeping for {} seconds".format(self.sleep_time))
        await asyncio.sleep(self.sleep_time)
        self.log("Done sleeping")
        return kwargs


class TestChainOperator(AsyncioTestCase):

    def test_chain_dag(self):
        op3 = SleepOperator(name="SleeperTest2", sleep_time=2)
        op4 = SleepOperator(name="SleeperTest2", sleep_time=2)
        chain_dag = Chain(name="TestSequential", children=[op3, op4])
        result2 = self.loop.run_until_complete(chain_dag(inputs={'time': 10}))
        self.assertIsNotNone(result2)


if __name__ == '__main__':
    unittest.main()
