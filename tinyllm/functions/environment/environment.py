import datetime as dt
import json

from smartpy.utility.log_util import getLogger
from tinyllm.functions.util.helpers import get_openai_message
from tinyllm.function_stream import FunctionStream
from tinyllm.validator import Validator

logger = getLogger(__name__)


class TinyEnvironmentOutputValidator(Validator):
    type: str
    response: dict


class TinyEnvironment(FunctionStream):

    def __init__(self,
                 llm_store,
                 tool_store,
                 manager_llm: str,
                 manager_args: dict,
                 **kwargs):

        super().__init__(**kwargs)

        self.llm_store = llm_store
        self.tool_store = tool_store
        self.tool_store.trace = self.trace
        self.manager = self.llm_store.get_agent(llm=manager_llm,
                                                llm_args=manager_args,
                                                trace=self.trace)

    def initialize_stream(self):
        assistant_response = ""
        tool_call = {
            "name": None,
            "arguments": ""
        }
        msg_role = None

        return msg_role, assistant_response, tool_call

    async def run(self,
                  user_input):

        input_msg = get_openai_message(role='user',
                                       content=user_input)
        while True:
            async for msg in self.manager(tool_choice='auto',
                                          tools=self.tool_store.tools,
                                          **input_msg):
                yield msg

            # Agent decides to call a tool
            if msg['type'] == 'tool':
                tool_msg = await self.llm_store.tool_store.run_tool(**msg['completion'])
                await self.manager.memory(message=tool_msg)
            elif msg['type'] == 'completion':
                break
