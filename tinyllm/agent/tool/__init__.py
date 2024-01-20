from tinyllm.agent.tool.toolkit import Toolkit
from tinyllm.agent.tool.tools.code_interpreter import get_code_interpreter_tool
from tinyllm.agent.tool.tools.think_plan import get_think_and_plan_tool
from tinyllm.agent.tool.tools.wikipedia import get_wikipedia_summary_tool


def tinyllm_toolkit():
    return Toolkit(
    name='Toolkit',
    tools=[
        get_think_and_plan_tool(),
        get_code_interpreter_tool(),
        get_wikipedia_summary_tool(),
    ],
)

