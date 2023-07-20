import asyncio

from tinyllm.chain import ParallelChain, SequentialChain
from tinyllm.operator import Operator


class SleepOperator(Operator):
    def __init__(self, name: str, sleep_time: float):
        super().__init__(name)
        self.sleep_time = sleep_time

    async def get_output(self, *args, **kwargs):
        print(kwargs)
        await asyncio.sleep(self.sleep_time)
        return kwargs


async def main():
    op1 = SleepOperator("Op1", 30)
    op2 = SleepOperator("Op2", 30)
    parallel_chain = ParallelChain("TestParallel", [op1, op2])
    sequential_chain = SequentialChain("TestParallel", [op1, op2])

    result = await parallel_chain([{}, {}])
    result2 = await sequential_chain({'time': 10})
    print(result)
    asyncio.run(main())


