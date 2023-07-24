import unittest
import asyncio

from env_util.environment import openagents_env
from tinyllm.functions.chain import Chain
from tinyllm.config import AppConfig
from tinyllm.functions.parallel import Parallel
from tinyllm.functions.function import Function


class SleepOperator(Function):
    def __init__(self, sleep_time: float, **kwargs):
        super().__init__(**kwargs)
        self.sleep_time = sleep_time

    async def get_output(self, *args, **kwargs):
        self.log("Sleeping for {} seconds".format(self.sleep_time))
        await asyncio.sleep(self.sleep_time)
        self.log("Done sleeping")
        return kwargs


class TestSleepOperator(unittest.TestCase):
    def setUp(self):
        self.loop = asyncio.get_event_loop()

    def test_parallel_dag(self):
        APP_CONFIG = AppConfig()

        APP_CONFIG.set_provider('openai', {
            "api_key":
                openagents_env.configs.OPENAI_API_KEY
        })

        op1 = SleepOperator(name="SleeperTest1", sleep_time=5)
        op2 = SleepOperator(name="SleeperTest1", sleep_time=5)
        parallel_dag = Parallel(name="TestParallel", children=[op1, op2])
        result1 = self.loop.run_until_complete(parallel_dag(inputs=[{'time': 10},
                                                                   {'time': 10}]))
        self.assertIsNotNone(result1)

    def test_chain_dag(self):
        APP_CONFIG = AppConfig()

        APP_CONFIG.set_provider('openai', {
            "api_key":
                openagents_env.configs.OPENAI_API_KEY
        })

        op3 = SleepOperator(name="SleeperTest2", sleep_time=5)
        op4 = SleepOperator(name="SleeperTest2", sleep_time=5)
        chain_dag = Chain(name="TestSequential", children=[op3, op4])
        result2 = self.loop.run_until_complete(chain_dag(inputs={'time': 10}))
        self.assertIsNotNone(result2)


if __name__ == '__main__':
    unittest.main()