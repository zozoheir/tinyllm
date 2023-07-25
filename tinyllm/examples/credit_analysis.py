import asyncio
import os

import openai

from tinyllm.functions.chain import Chain
from tinyllm.functions.decision import Decision
from tinyllm.functions.llms.openai_chat import OpenAIChat
from tinyllm.functions.parallel import Concurrent
from tinyllm.functions.function import Function
from tinyllm.functions.prompts.openai_chat.system import OpenAISystemMessage
from tinyllm.functions.prompts.template import OpenAIPromptTemplate
from tinyllm.functions.prompts.user_input import OpenAIUserMessage
from tinyllm.graph import graph_chain

openai.api_key = os.getenv("OPENAI_API_KEY")

good_loan_application_example = """
The loan application showcases a commendable financial profile with an excellent credit history. The applicant's credit score
demonstrates a history of timely payments, responsible credit usage, and a low utilization rate, reflecting a consistent
track record of financial prudence. Moreover, the applicant's income documentation reveals a stable employment history
with a steady and substantial income stream. The debt-to-income ratio is well within the acceptable range, indicating a
manageable level of existing debt. Overall, the applicant's solid credit standing and stable financial situation make
this loan application an appealing opportunity for potential lenders.
"""


# Initialize a Loan Classifier OpenAI Prompt template
openai_chat = OpenAIChat(name='OpenAI-GPT model',
                         llm_name='gpt-3.5-turbo',
                         temperature=0,
                         n=1)
loan_classifier_role = OpenAISystemMessage(name="Role",
                                           content="You will be provided with a loan application."
                                                   "Your role is to classify if as as good or bad. Your output should be one one of these 2 words:[good, bad]")

loan_classifier_template = OpenAIPromptTemplate(name="Loan Classifier Template",
                                                sections=[
                                                    loan_classifier_role,
                                                    OpenAIUserMessage(name="User Message"),
                                                ])

# Initialize a Loan Classifier OpenAI Prompt template
async def classify_loan_application(**kwargs):
    loan_application = kwargs.get('loan_application')
    messages = await loan_classifier_template(message=loan_application)
    chat_response = await openai_chat(**messages)
    print(f"Credit classification: {chat_response}")
    return {'decision': chat_response}

loan_classifier = Decision(
    name="Loan classifier",
    choices=['good', 'bad'],
    run_function=classify_loan_application,
    verbose=True
)

async def send_email(**kwargs):
    print("Sending email notification...")
    return {'success': True}

email_notification = Function(
    name="Email notification",
    run_function=send_email
)

async def background_check(**kwargs):
    print("Performing background check")
    return {'background_check': 'Completed'}

bg_check = Function(
    name="Background check",
    run_function=background_check,
)

async def main():

    credit_analysis_chain = Chain(name="Loan application processing",
                                  children=[
                                      loan_classifier,
                                      Concurrent(name="On good credit",
                                                 children=[email_notification,
                                                           bg_check])],
                                  verbose=True)

    result = await credit_analysis_chain(loan_application=good_loan_application_example)
    graph_chain(credit_analysis_chain)



if __name__ == '__main__':
    asyncio.run(main())
