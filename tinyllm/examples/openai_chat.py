# Initialize a Loan Classifier OpenAI Prompt template
import asyncio
import os

import openai
from tinyllm.functions.llms.openai.openai_chat import OpenAIChat

openai.api_key = os.environ['OPENAI_API_KEY']

async def main():
    openai_chat = OpenAIChat(name='OpenAI Chat model',
                             llm_name='gpt-3.5-turbo',
                             temperature=0,
                             n=1,
                             verbose=True)

    chat_response = await openai_chat(message='Hi how are you?')


if __name__ == '__main__':
    asyncio.run(main())
