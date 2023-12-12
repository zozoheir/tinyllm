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
                                          tools=self.toolkit.as_dicts()):
                yield msg

            # Agent decides to call a tool
            if msg['status'] == 'success':
                msg_output = msg['output']
                if msg_output['type'] == 'tool':
                    # TODO When ready, implement parallel function calls
                    api_tool_call = msg_output['delta']['tool_calls'][0]
                    msg_output['delta'].pop('function_call')

                    # Create tool call message memory
                    json_tool_call = {
                        'name': msg_output['completion']['name'],
                        'arguments': msg_output['completion']['arguments']
                    }
                    api_tool_call['function'] = json_tool_call
                    msg_output['delta']['content'] = ''
                    res = await self.manager.memorize(message=msg_output['delta'])

                    # Create tool result message memory
                    tool_results = await self.toolkit(
                        tool_calls=[{
                            'name': api_tool_call['function']['name'],
                            'arguments': json.loads(api_tool_call['function']['arguments'])
                        }],
                        trace=self.trace)
                    tool_result = tool_results['output']['tool_results'][0]
                    function_call_msg = get_openai_message(
                        name=tool_result['name'],
                        role='tool',
                        content=tool_result['content'],
                        tool_call_id=api_tool_call['id']
                    )
                    input_openai_msg = function_call_msg

                elif msg_output['type'] == 'completion':
                    break
            else:
                raise (Exception(msg))
