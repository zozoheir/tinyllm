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
        llm = llm_store.get_llm(
            name='Tinyllm manager',
            llm_library=LLMs.LITE_LLM,
            is_traced=False,
            debug=False
        )
        tiny_agent = Agent(
            system_role="You are a helpful assistant",
            name='Test: agent',
            llm=llm,
            example_manager=ExampleManager(),
            toolkit=toolkit,
            memory=BufferMemory(name='Agent memory', is_traced=False),
            evaluators=[
                AnswerCorrectnessEvaluator(
                    name="Eval: correct user info",
                    is_traced=False,
                ),
            ],
            debug=True,
            is_traced=True
        )

        # Run the asynchronous test
        result = self.loop.run_until_complete(tiny_agent(user_input="What is the user's birthday?"))
        first_choice_message = result['output']['response']['choices'][0]['message']
        # Verify the last message in the list
        self.assertEqual(result['status'], 'success')
        self.assertTrue('january 1st' in first_choice_message['content'].lower())


# This allows the test to be run standalone
if __name__ == '__main__':
    unittest.main()
