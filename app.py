import asyncio

from env_util.environment import openagents_env
from tinyllm.chain import Chain
from tinyllm.config import AppConfig
from tinyllm.logger import get_logger
from tinyllm.parallel import Parallel
from tinyllm.types import Functions
from tinyllm.function import Function


class SleepOperator(Function):
    def __init__(self, sleep_time: float, **kwargs):
        super().__init__(type=Functions.BASE,
                         **kwargs)
        self.sleep_time = sleep_time

    async def get_output(self, *args, **kwargs):
        self.log("Sleeping for {} seconds".format(self.sleep_time))
        await asyncio.sleep(self.sleep_time)
        self.log("Done sleeping")
        return kwargs


async def main():
    APP_CONFIG = AppConfig()

    APP_CONFIG.set_provider('openai', {
        "api_key":
            openagents_env.configs.OPENAI_API_KEY
    })

    op1 = SleepOperator(name="SleeperTest1", sleep_time=5)
    op2 = SleepOperator(name="SleeperTest1", sleep_time=5)
    parallel_dag = Parallel(name="TestParallel", children=[op1, op2])
    result1 = await parallel_dag(inputs=[{'time': 10},
                                         {'time': 10}])

    op3 = SleepOperator(name="SleeperTest2", sleep_time=5)
    op4 = SleepOperator(name="SleeperTest2", sleep_time=5)
    chain_dag = Chain(name="TestSequential", children=[op3, op4])
    result2 = await chain_dag(inputs={'time': 10})


if __name__ == '__main__':
    asyncio.run(main())
