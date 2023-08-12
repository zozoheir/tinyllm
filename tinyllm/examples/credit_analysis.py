import asyncio

from tinyllm.functions.chain import Chain
from tinyllm.functions.decision import Decision
from tinyllm.functions.llms.open_ai.openai_chat import OpenAIChat
from tinyllm.functions.concurrent import Concurrent
from tinyllm.functions.function import Function
from tinyllm.functions.llms.open_ai.openai_prompt_template import OpenAIPromptTemplate

good_loan_application_example = """
The loan application showcases a commendable financial profile with an excellent credit history. The applicant's credit score
demonstrates a history of timely payments, responsible credit usage, and a low utilization rate, reflecting a consistent
track record of financial prudence. Moreover, the applicant's income documentation reveals a stable employment history
with a steady and substantial income stream. The debt-to-income ratio is well within the acceptable range, indicating a
manageable level of existing debt. Overall, the applicant's solid credit standing and stable financial situation make
this loan application an appealing opportunity for potential lenders.
"""

loan_classifier_prompt_template = OpenAIPromptTemplate(
    name="KG Extractor prompt template",
    system_role="Your role is to classify if as as good or bad. Your output should be one one of these 2 words:[good, bad]",
)

openai_chat = OpenAIChat(name='OpenAI-GPT model',
                         llm_name='gpt-3.5-turbo',
                         temperature=0,
                         n=1,
                         prompt_template=loan_classifier_prompt_template)


# Loan classifier LLM
async def classify_loan_application(**kwargs):
    loan_application = kwargs.get('loan_application')
    chat_response = await openai_chat(message=loan_application)
    print(f"Credit classification: {chat_response}")
    return {'decision': chat_response['response']}


loan_classifier = Decision(
    name="Decision: Loan classifier",
    choices=['good', 'bad'],
    run_function=classify_loan_application,
    is_traced=True
)


# Email notification
async def send_email(**kwargs):
    print("Sending email notification...")
    return {'success': True}


email_notification = Function(
    name="Email notification",
    run_function=send_email
)


# Background get_check_results
async def background_check(**kwargs):
    print("Performing background get_check_results")
    return {'background_check': 'Completed'}


bg_check = Function(
    name="Background get_check_results",
    run_function=background_check,
)


# Chain
async def main():
    credit_analysis_chain = Chain(
        name="Chain: Loan application",
        children=[
            loan_classifier,
            Concurrent(name="Concurrent: On good credit",
                       children=[
                           email_notification,
                           bg_check])
        ],
        is_traced=True)

    result = await credit_analysis_chain(loan_application=good_loan_application_example)


if __name__ == '__main__':
    asyncio.run(main())
