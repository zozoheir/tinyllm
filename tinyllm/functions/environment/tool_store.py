import datetime as dt

from langfuse.model import CreateSpan, UpdateSpan

from tinyllm.functions.util.helpers import get_openai_message


class ToolStore:

    def __init__(self,
                 tools,
                 tools_callables,
                 trace=None):
        self.tools = tools
        self.tools_callables = tools_callables
        self.trace = trace

    async def run_tool(self,
                       tool_name,
                       tool_arguments):
        if self.trace:
            span = self.trace.span(
                CreateSpan(
                    name="tool: " + tool_name,
                    input=tool_arguments,
                    startTime=dt.datetime.now()
                )
            )
        tool_response = self.tools_callables[tool_name](**tool_arguments)

        if self.trace:
            span.update(
                UpdateSpan(
                    output=tool_response,
                    endTime=dt.datetime.now())
            )
        return get_openai_message(role='function', content=tool_response, name=tool_name)
