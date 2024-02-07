import unittest

from tinyllm.agent.agent import Agent
from tinyllm.agent.tool import tinyllm_toolkit
from tinyllm.eval.evaluator import Evaluator
from tinyllm.llms.llm_store import LLMStore
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


llm_store = LLMStore()


# Define the test class

class TestStreamingAgent(AsyncioTestCase):

    def test_wiki_tool(self):
        tiny_agent = Agent(
            system_role="You are a helpful assistant",
            llm=llm_store.default_llm,
            toolkit=tinyllm_toolkit(),
            user_id='test_user',
            session_id='test_session',
        )
        # Run the asynchronous test
        result = self.loop.run_until_complete(tiny_agent(user_input="What does wiki say about Morocco"))
        self.assertTrue(result['status'] == 'success')


    def test_fibonacci_code(self):
        tiny_agent = Agent(
            system_role="You are a helpful assistant",
            llm=llm_store.default_llm,
            toolkit=tinyllm_toolkit(),
            user_id='test_user',
            session_id='test_session',
        )
        # Run the asynchronous test
        result = self.loop.run_until_complete(tiny_agent(user_input="Use python to give me the 5th fibonacci number"))
        self.assertTrue(result['status'] == 'success')
        if result['status'] == 'success':
            msg_content = result['output']['response']['choices'][0]['message']['content']
            self.assertTrue('5' in str(msg_content))

    def test_multi_tool(self):
        tiny_agent = Agent(
            name="Test: Agent multi Tool",
            system_role="You are a helpful assistant",
            llm=llm_store.default_llm,
            toolkit=tinyllm_toolkit(),
            user_id='test_user',
            session_id='test_session',
        )
        # Run the asynchronous test
        query = """Plan then execute this task for me: I need to multiply the population of Morocco by the population of
         Senegal, then square that number by Elon Musk's age"""
        result = self.loop.run_until_complete(tiny_agent(user_input=query,
                                                         model='gpt-3.5-turbo')) # Parallel call is not handled yet
        self.assertTrue(result['status'] == 'success')


# This allows the test to be run standalone
if __name__ == '__main__':
    unittest.main()
