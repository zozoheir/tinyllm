# Initialize a Loan Classifier OpenAI Prompt template
import asyncio

from tinyllm.functions.llms.openai_chat import OpenAIChat
from tinyllm.functions.prompts.openai_chat.system import OpenAISystemMessage
from tinyllm.functions.prompts.template import OpenAIPromptTemplate
from tinyllm.functions.prompts.user_input import OpenAIUserMessage



async def main():
    openai_chat = OpenAIChat(name='OpenAI Chat model',
                             llm_name='gpt-3.5-turbo',
                             temperature=0,
                             n=1,
                             verbose=True)

    loan_classifier_role = OpenAISystemMessage(name="Role",
                                               content="You will be provided with a loan application."
                                                       "Your role is to classify if as as good or bad. Your output should be one one of these 2 words:[good, bad]")

    loan_classifier_template = OpenAIPromptTemplate(name="Loan Classifier Template",
                                                    sections=[
                                                        loan_classifier_role,
                                                        OpenAIUserMessage(name="User Message"),
                                                    ])
    messages = await loan_classifier_template(message="Hii")

    chat_response = await openai_chat(**messages)


    response = await openai_chat(**messages)
    print(response)

if __name__ == '__main__':
    asyncio.run(main())