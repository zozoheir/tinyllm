import asyncio

from tinyllm.chain import ParallelChain, SequentialChain
from tinyllm.operator import Operator


class SleepOperator(Operator):
    def __init__(self, sleep_time: float, **kwargs):
        super().__init__(**kwargs)
        self.sleep_time = sleep_time

    async def get_output(self, *args, **kwargs):
        self.log("Sleeping for {} seconds".format(self.sleep_time))
        await asyncio.sleep(self.sleep_time)
        self.log("Done sleeping")
        return kwargs


async def main():

    op1 = SleepOperator(name="SleeperTest1", sleep_time=5)
    op2 = SleepOperator(name="SleeperTest1", sleep_time=5)
    parallel_chain = ParallelChain(name="TestParallel", children=[op1, op2])
    result = await parallel_chain([{}, {}])


    op3 = SleepOperator(name="SleeperTest2", sleep_time=5)
    op4 = SleepOperator(name="SleeperTest2", sleep_time=5)
    sequential_chain = SequentialChain(name="TestSequential", children=[op3, op4])
    result2 = await sequential_chain(time=10)
    print(result)


if __name__ == '__main__':
    asyncio.run(main())