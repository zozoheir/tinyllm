import unittest
from typing import Any, Optional

from pydantic import BaseModel, Field

from tinyllm.agent.agent import Agent
from tinyllm.agent.tool import tinyllm_toolkit
from tinyllm.eval.evaluator import Evaluator
from tinyllm.llms.lite_llm import LiteLLM
from tinyllm.tests.base import AsyncioTestCase
from tinyllm.util.message import Text, UserMessage


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


# Define the test class

class TestAgent(AsyncioTestCase):

    def test_json_output(self):
        class Person(BaseModel):
            name: str = Field(..., description='Name of the person')
            age: int = Field(..., description='Age of the person')
            note: Optional[Any]

        class RiskScoreOutput(BaseModel):
            risk_score: float = Field(..., description='Confidence level of the trade idea between 0 and 100')
            person: Person

        tiny_agent = Agent(
            name='Test: Agent JSON output',
            system_role="You are a Credit Risk Analyst. Respond with a risk score based on the provided customer data",
            json_pydantic_model=RiskScoreOutput
        )
        # Run the asynchronous test
        result = self.loop.run_until_complete(tiny_agent(content="Johny Vargas, 29yo, the customer has missed 99% of his bill payments in the last year"))
        self.assertTrue(result['status'] == 'success')
        self.assertTrue(result['output']['response'].get('risk_score') is not None)


    def test_wiki_tool(self):
        tiny_agent = Agent(
            name='Test: Agent Wiki Tool',
            llm=LiteLLM(),
            toolkit=tinyllm_toolkit(),
            user_id='test_user',
            session_id='test_session',
        )
        # Run the asynchronous test
        result = self.loop.run_until_complete(tiny_agent(content="What does wiki say about Morocco"))
        self.assertTrue(result['status'] == 'success')


    def test_multi_tool(self):
        tiny_agent = Agent(
            name="Test: Agent Multi Tool",
            toolkit=tinyllm_toolkit(),
        )
        # Run the asynchronous test
        query = """Plan then execute this task for me: I need to multiply the population of Morocco by the population of
         Senegal, then square that number by Elon Musk's age"""
        result = self.loop.run_until_complete(tiny_agent(content=query,
                                                         model='gpt-4'))  # Parallel call is not handled yet
        self.assertTrue(result['status'] == 'success')



# This allows the test to be run standalone
if __name__ == '__main__':
    unittest.main()
