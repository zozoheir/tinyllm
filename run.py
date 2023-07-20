import asyncio

from tinyllm import Chains
from tinyllm.chain import Chain
from tinyllm.operator import Operator


class EchoOperator(Operator):
    async def get_output(self, **kwargs):
        return {"id": self.id, "name": self.name, "output": "test output"}

# Create child operators
child1 = EchoOperator(name="Child Operator")
child2 = EchoOperator(name="Child Operator")
child3 = EchoOperator(name="Child Operator")

# Create a parent operator that executes the children in parallel
parent = Chain(name="ParentChain",
               type=Chains.SEQUENTIAL,
               children=[child1, child2, child3])

# Run the parent operator
result = asyncio.run(parent(data="Hello"))
print(result)
