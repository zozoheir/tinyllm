"""
VISUALIZATION OF THE AGENT STREAM EXAMPLE

https://us.cloud.langfuse.com/project/cloz2bp020000l008kg9ujywd/traces/e5d46730-8528-4bd1-935d-ab949904f80d
"""

import asyncio

from tinyllm.agent.agent import Agent
from tinyllm.agent.tool import Toolkit
from tinyllm.agent.tool.tool import Tool
from tinyllm.eval.evaluator import Evaluator
from tinyllm.memory.memory import BufferMemory

loop = asyncio.get_event_loop()

class AnswerCorrectnessEvaluator(Evaluator):

    async def run(self, **kwargs):
        completion = kwargs['response']['choices'][0]['message']['content']
        evals = {
            "evals": {
                "correct_answer": 1 if 'january 1st' in completion.lower() else 0
            },
            "metadata": {}
        }

        return evals


def get_user_property(asked_property):
    if asked_property == "name":
        return "Elias"
    elif asked_property == "birthday":
        return "January 1st"


tools = [
    Tool(
        name="get_user_property",
        description="This is the tool you use retrieve ANY information about the user. Use it to answer questions about his birthday and any personal info",
        python_lambda=get_user_property,
        parameters={
            "type": "object",
            "properties": {
                "asked_property": {
                    "type": "string",
                    "enum": ["birthday", "name"],
                    "description": "The specific property the user asked about",
                },
            },
            "required": ["asked_property"],
        },
    )
]
toolkit = Toolkit(
    name='Toolkit',
    tools=tools,
)

async def run_agent():
    tiny_agent = Agent(system_role="You are a helpful agent that can answer questions about the user's profile using available tools.",
                       toolkit=toolkit,
                       memory=BufferMemory(),
                       run_evaluators=[
                           AnswerCorrectnessEvaluator(),
                       ])

    result = await tiny_agent(content="What is the user's birthday?")
    print(result)

result = loop.run_until_complete(run_agent())
