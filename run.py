import asyncio

from tinyllm.chain import ParallelChain
from tinyllm.operator import Operator


class SleepOperator(Operator):
    def __init__(self, name: str, sleep_time: float):
        super().__init__(name)
        self.sleep_time = sleep_time

    async def get_output(self, *args, **kwargs):
        print(kwargs)
        await asyncio.sleep(self.sleep_time)
        return {self.name: f"Slept for {self.sleep_time} seconds"}

async def main():
    op1 = SleepOperator("Op1", 2)
    op2 = SleepOperator("Op2", 5)
    chain = ParallelChain("TestParallel", [op1, op2])

    result = await chain([{}, {}])
    print(result)

if __name__ == "__main__":
    asyncio.run(main())