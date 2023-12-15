import asyncio
import unittest

from tests.base import AsyncioTestCase
from tinyllm.functions.agent.agent import Agent
from tinyllm.functions.agent.agent_stream import AgentStream
from tinyllm.functions.agent.toolkit import Toolkit
from tinyllm.functions.llms.llm_store import LLMStore, LLMs

from tinyllm.functions.agent.tool import Tool
from tinyllm.functions.eval.evaluator import Evaluator
from tinyllm.functions.memory.memory import Memory, BufferMemory


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
    is_traced=False,
)

llm_store = LLMStore()


# Define the test class
class TestStreamingAgent(AsyncioTestCase):

    def test_agent(self):
        manager_llm = llm_store.get_llm_function(
            llm_library=LLMs.LITE_LLM,
            system_role="You are a helpful agent that can answer questions about the user's profile using available tools.",
            name='Tinyllm manager',
            is_traced=False,
            debug=False
        )
        tiny_agent = Agent(name='Test: agent',
                           manager_llm=manager_llm,
                           toolkit=toolkit,
                           memory=BufferMemory(name='Agent memory', is_traced=False),
                           evaluators=[
                               AnswerCorrectnessEvaluator(
                                   name="Eval: correct user info",
                                   is_traced=False,
                               ),
                           ],
                           debug=True)

        # Run the asynchronous test
        result = self.loop.run_until_complete(tiny_agent(user_input="What is the user's birthday?"))
        first_choice_message = result['output']['response']['choices'][0]['message']
        # Verify the last message in the list
        self.assertEqual(result['status'], 'success')
        self.assertTrue('january 1st' in first_choice_message['content'].lower())


# This allows the test to be run standalone
if __name__ == '__main__':
    unittest.main()
