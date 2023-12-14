import asyncio
from typing import List, Dict

from tinyllm.function import Function
from tinyllm.functions.agent.tool import Tool
from tinyllm.util.tracing.span import langfuse_span
from tinyllm.validator import Validator


class ToolkitInputValidator(Validator):
    tool_calls: List[Dict]


class ToolkitOutputValidator(Validator):
    tool_results: List[Dict]


class Toolkit(Function):

    def __init__(self,
                 tools: List[Tool],
                 **kwargs):
        super().__init__(
            input_validator=ToolkitInputValidator,
            **kwargs)
        self.tools = tools
        # We set the tools as attributes to pass the langfuse trace to the tools

    @langfuse_span(name='Tookit', input_key='tool_calls')
    async def run(self,
                  **kwargs):
        tasks = []

        for tool_call in kwargs['tool_calls']:
            name = tool_call['name']
            arguments = tool_call['arguments']
            tasks.append(self.run_tool(name=name,
                                       arguments=arguments,
                                       parent_observation=self.parent_observation))
        results = await asyncio.gather(*tasks)
        tool_results = [result['output']['response'] for result in results]
        return {'tool_results': tool_results}


    async def run_tool(self,
                       name: str,
                       arguments: Dict,
                       **kwargs):
        tool = [tool for tool in self.tools if tool.name == name][0]
        tool_result = await tool(arguments=arguments,
                                 **kwargs)
        return tool_result

    def as_dict_list(self):
        return [tool.as_dict() for tool in self.tools]
