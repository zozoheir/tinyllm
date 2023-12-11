import json
from typing import List

from smartpy.utility.log_util import getLogger
from tinyllm.function import Function

from tinyllm.functions.agent.tool import Tool
from tinyllm.functions.agent.toolkit import Toolkit
from tinyllm.functions.helpers import get_openai_message
from tinyllm.function_stream import FunctionStream
from tinyllm.validator import Validator

logger = getLogger(__name__)


class TinyEnvironmentInitValidator(Validator):
    manager_function: Function
    planner_function: Function
    toolkit: Toolkit


class TinyEnvironmentOutputValidator(Validator):
    type: str
    response: dict


class Agent(FunctionStream):

    def __init__(self,
                 manager_function: Function,
                 toolkit: Toolkit,
                 **kwargs):
        super().__init__(**kwargs)
        self.toolkit = toolkit
        self.manager = manager_function

    async def run(self,
                  user_input):

        input_openai_msg = get_openai_message(role='user',
                                       content=user_input)
        while True:

            async for msg in self.manager(message=input_openai_msg,
                                          tools=self.toolkit.as_dicts(),
                                          tool_choice='auto'):
                yield msg

            # Agent decides to call a tool
            if msg['status'] == 'success':
                msg_output = msg['output']
                if msg_output['type'] == 'tool':
                    name = msg_output['completion']['name']
                    arguments = json.loads(msg_output['completion']['arguments'])
                    tool_calls = [{'name': name,
                                   'arguments': arguments}]
                    tool_results = await self.toolkit(tool_calls=tool_calls)
                    # TODO Add multi tool output for parallel calls
                    input_openai_msg = tool_results['output']['tool_results'][0]
                elif msg_output['type'] == 'completion':
                    break
            else:
                raise (Exception(msg))
