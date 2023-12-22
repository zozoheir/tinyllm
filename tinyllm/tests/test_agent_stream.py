import unittest

from tinyllm.tests.base import AsyncioTestCase
from tinyllm.agent.agent_stream import AgentStream
from tinyllm.agent.tool import Tool
from tinyllm.agent.toolkit import Toolkit
from tinyllm.eval.evaluator import Evaluator
from tinyllm.examples.example_manager import ExampleManager
from tinyllm.llms.llm_store import LLMStore, LLMs
from tinyllm.memory.memory import BufferMemory


class AnswerCorrectnessEvaluator(Evaluator):
    async def run(self, **kwargs):
        completion = kwargs['output']['output']['completion']

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
    tools=tools
)

llm_store = LLMStore()


# Define the test class
class TestStreamingAgent(AsyncioTestCase):

    def test_agent_stream(self):
        llm = llm_store.get_llm(
            llm_library=LLMs.LITE_LLM_STREAM,
            name='Tinyllm manager',
        )

        tiny_agent = AgentStream(
            system_role="You are a helpful assistant",
            name='Test: agent stream',
            llm=llm,
            example_manager=ExampleManager(),
            toolkit=toolkit,
            memory=BufferMemory(name='Agent memory'),
            run_evaluators=[
                AnswerCorrectnessEvaluator(
                    name="Eval: correct user info",
                ),
            ],
        )

        async def async_test():
            msgs = []
            async for message in tiny_agent(user_input="What is the user's birthday?"):
                msgs.append(message)
            return msgs

        # Run the asynchronous test
        result = self.loop.run_until_complete(async_test())
        # Verify the last message in the list
        self.assertEqual(result[-1]['status'], 'success', "The last message status should be 'success'")
        self.assertTrue("january 1st" in result[-1]['output']['completion'].lower())


# This allows the test to be run standalone
if __name__ == '__main__':
    unittest.main()
