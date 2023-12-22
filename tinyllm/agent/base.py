from typing import Union

from tinyllm.agent.toolkit import Toolkit
from tinyllm.examples.example_manager import ExampleManager
from tinyllm.function import Function
from tinyllm.function_stream import FunctionStream
from tinyllm.memory.memory import BufferMemory
from tinyllm.prompt_manager import PromptManager


class AgentBase:

    def __init__(self,
                 llm: Union[FunctionStream, Function],
                 toolkit: Toolkit,
                 system_role: str = "You are a helpful assistant",
                 memory=BufferMemory(name='Agent memory'),
                 example_manager=ExampleManager(),
                 **kwargs):
        self.toolkit = toolkit
        self.llm = llm
        self.memory = memory
        self.example_manager = example_manager
        self.prompt_manager = PromptManager(
            system_role=system_role,
            example_manager=example_manager,
            memory=memory,
        )
