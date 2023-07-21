import asyncio

from tinyllm.collection import ParallelChain, SequentialChain
from tinyllm.types import Operators
from tinyllm.operator import Operator


class SleepOperator(Operator):
    def __init__(self, sleep_time: float, **kwargs):
        super().__init__(operator_type=Operators.OPERATOR, **kwargs)
        self.sleep_time = sleep_time

    async def get_output(self, *args, **kwargs):
        self.log("Sleeping for {} seconds".format(self.sleep_time))
        await asyncio.sleep(self.sleep_time)
        self.log("Done sleeping")
        return kwargs


import logging

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(name)s %(message)s')


async def main():
    op1 = SleepOperator(name="SleeperTest1", sleep_time=5)
    op2 = SleepOperator(name="SleeperTest1", sleep_time=5)
    parallel_chain = ParallelChain(name="TestParallel", children=[op1, op2])
    result1 = await parallel_chain(inputs=[{'time': 10},
                                           {'time': 10}])

    op3 = SleepOperator(name="SleeperTest2", sleep_time=5)
    op4 = SleepOperator(name="SleeperTest2", sleep_time=5)
    sequential_chain = SequentialChain(name="TestSequential", children=[op3, op4])
    result2 = await sequential_chain(time=10)


if __name__ == '__main__':
    asyncio.run(main())
