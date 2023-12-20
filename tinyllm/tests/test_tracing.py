import asyncio
import unittest

from tinyllm.agent.agent import Agent
from tinyllm.agent.tool import Tool
from tinyllm.agent.toolkit import Toolkit
from tinyllm.eval.evaluator import Evaluator
from tinyllm.examples.example_manager import ExampleManager
from tinyllm.llms.llm_store import LLMStore, LLMs
from tinyllm.memory.memory import BufferMemory
from tinyllm.tests.base import AsyncioTestCase


class AnswerCorrectnessEvaluator(Evaluator):

    async def run(self, **kwargs):
        completion = kwargs['output']['response']['choices'][0]['message']['content']
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

def get_tools():
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
            is_traced=False,
        )
    ]
    toolkit = Toolkit(
        name='Toolkit',
        tools=tools,
    )
    return toolkit


llm_store = LLMStore()


# Define the test class
class TestStreamingAgent(AsyncioTestCase):

    def test_agent(self):
        tiny_agent1 = Agent(
            system_role="You are a helpful assistant",
            name='Agent 1',
            example_manager=ExampleManager(),
            toolkit=get_tools(),
            memory=BufferMemory(name='Agent memory'),
        )
        tasks = [tiny_agent1(user_input="What is the user's birthday?")]
        res = self.loop.run_until_complete(asyncio.gather(*tasks))


    def test_concurrent_agents(self):
        tiny_agent1 = Agent(
            system_role="You are a helpful assistant",
            name='Agent 1',
            example_manager=ExampleManager(),
            toolkit=get_tools(),
            memory=BufferMemory(name='Agent memory'),
        )
        tiny_agent2 = Agent(
            system_role="You are a helpful assistant",
            name='Agent 2',
            example_manager=ExampleManager(),
            toolkit=get_tools(),
            memory=BufferMemory(name='Agent memory'),
        )

        # Run the asynchronous tasks concurrently
        tasks = [tiny_agent1(user_input="What is the user's birthday?"),
                 tiny_agent2(user_input="What is the user's birthday?")]

        res = self.loop.run_until_complete(asyncio.gather(*tasks))



# This allows the test to be run standalone
if __name__ == '__main__':
    unittest.main()
