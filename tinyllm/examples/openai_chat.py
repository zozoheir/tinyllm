# Initialize a Loan Classifier OpenAI Prompt template
import asyncio
import os

import openai

from tinyllm import APP
from tinyllm.functions.llms.openai.openai_chat import OpenAIChat

openai.api_key = os.environ['OPENAI_API_KEY']
APP.connect_graph_db(host=os.environ['TINYLLM_DB_HOST'],
                     port=os.environ['TINYLLM_DB_PORT'],
                     user=os.environ['TINYLLM_DB_USER'],
                     password=os.environ['TINYLLM_DB_PASSWORD'])


async def main():
    openai_chat = OpenAIChat(name='OpenAI Chat model',
                             llm_name='gpt-3.5-turbo',
                             temperature=0,
                             n=1,
                             verbose=True)

    chat_response = await openai_chat(message='Hi how are you?')


if __name__ == '__main__':
    asyncio.run(main())
