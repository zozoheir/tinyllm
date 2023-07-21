import asyncio

from tinyllm.chain import Chain
from tinyllm.parallel import Parallel
from tinyllm.types import Functions
from tinyllm.function import Function


import logging

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(name)s %(message)s')



class SleepOperator(Function):
    def __init__(self, sleep_time: float, **kwargs):
        super().__init__(function_type=Functions.OPERATOR, **kwargs)
        self.sleep_time = sleep_time

    async def get_output(self, *args, **kwargs):
        self.log("Sleeping for {} seconds".format(self.sleep_time))
        await asyncio.sleep(self.sleep_time)
        self.log("Done sleeping")
        return kwargs

async def main():
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
