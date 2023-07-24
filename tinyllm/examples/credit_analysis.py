import asyncio

from tinyllm.config import App, APP_CONFIG
from tinyllm.functions.chain import Chain
from tinyllm.functions.decision import Decision
from tinyllm.functions.parallel import Parallel
from tinyllm.functions.function import Function
from tinyllm.logger import get_logger
from tinyllm.stores.openai import OpenAILLM

openai_llm = OpenAILLM(
    llm_name='gpt-3.5-turbo',
    llm_params={'temparature': 0.9},
)


class CreditBucketDecision(Decision):
    def __init__(self, choices, **kwargs):
        super().__init__(choices=choices,
                         **kwargs)

    async def run(self, **kwargs):
        text = kwargs.get('text')
        credit_class = 'good' if 'good' in text else 'bad'
        print(f"Credit score classification done on: {credit_class}")
        return {'decision': credit_class}


class EmailNotification(Function):

    async def run(self, **kwargs):
        print(f"Sending Email to management with credit score: {kwargs.get('decision')}")
        return {'success':True}


class FurtherAnalysis(Function):

    async def run(self, **kwargs):
        text = kwargs.get('text')
        print(f"Further analysis done on: {text}")
        return {'further_analysis': 'done'}


async def main():
    credit_decision = CreditBucketDecision(name="CreditClassification",
                                           choices=['good', 'bad'])
    email_notification = EmailNotification(name="EmailNotification")
    further_analysis = FurtherAnalysis(name="FurtherAnalysis")

    chain = Chain(name="CreditAnalysis",
                  children=[
                      credit_decision,
                      Parallel(name="CreditAnalysisParallel",
                               children=[email_notification,
                                         further_analysis])])

    result = await chain(text="This is a bad loan application")


if __name__ == '__main__':
    APP_CONFIG.set_logging('default', get_logger(name='default'))
    asyncio.run(main())
