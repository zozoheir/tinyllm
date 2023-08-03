# Initialize a Loan Classifier OpenAI Prompt template
import asyncio
import os

import openai
from tinyllm.functions.llms.open_ai.openai_chat import OpenAIChat
from tinyllm.functions.llms.open_ai.openai_prompt_template import OpenAIPromptTemplate

openai.api_key = os.environ['OPENAI_API_KEY']

prompt_template = OpenAIPromptTemplate(
    name="TinyLLM Agent Prompt Template",
)
openai_chat = OpenAIChat(name='OpenAI Chat model',
                         llm_name='gpt-3.5-turbo',
                         temperature=0,
                         verbose=True,
                         max_tokens=100,
                         prompt_template=prompt_template)


if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    response = loop.run_until_complete(openai_chat(message='Hi how are you?'))
    print(response)
