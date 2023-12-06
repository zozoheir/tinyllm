import datetime as dt
from tinyllm.functions.util.helpers import get_openai_message


class ToolStore:

    def __init__(self,
                 tools,
                 tools_callables,
                 llm_trace=None):
        self.tools = tools
        self.tools_callables = tools_callables
        self.llm_trace = llm_trace

    async def run_tool(self,
                       tool_name,
                       tool_arguments):
        self.llm_trace.create_span(
            name="tool: " + tool_name,
            input=tool_arguments,
            startTime=dt.datetime.now(),
        )
        tool_response = self.tools_callables[tool_name](**tool_arguments)
        self.llm_trace.update_span(
            output=tool_response,
            endTime=dt.datetime.now(),
        )
        return get_openai_message(role='function', content=tool_response, name=tool_name)

