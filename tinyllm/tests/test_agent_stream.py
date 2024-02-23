import unittest

from tinyllm.agent.tool.tool import Tool
from tinyllm.tests.base import AsyncioTestCase
from tinyllm.agent.agent_stream import AgentStream
from tinyllm.agent.tool import Toolkit, tinyllm_toolkit
from tinyllm.eval.evaluator import Evaluator
from tinyllm.memory.memory import BufferMemory
from tinyllm.util.helpers import get_openai_message


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



# Define the test class
class TestStreamingAgent(AsyncioTestCase):

    def test_tool_call(self):

        tiny_agent = AgentStream(
            name="Test: Agent Stream tools",
            toolkit=tinyllm_toolkit(),
            user_id='test_user',
            session_id='test_session',
            run_evaluators=[
                AnswerCorrectnessEvaluator(
                    name="Eval: correct user info",
                ),
            ],
        )

        async def async_test():
            msgs = []
            async for message in tiny_agent(content="What is the 5th Fibonacci number?"):
                msgs.append(message)
            return msgs

        # Run the asynchronous test
        result = self.loop.run_until_complete(async_test())
        # Verify the last message in the list
        self.assertEqual(result[-1]['status'], 'success', "The last message status should be 'success'")


# This allows the test to be run standalone
if __name__ == '__main__':
    unittest.main()
