import asyncio

from tinyllm.functions.llms.openai_chat import OpenAIChat
from tinyllm.functions.prompts.openai_chat.system import OpenAISystemMessage
from tinyllm.functions.prompts.template import OpenAIPromptTemplate
from tinyllm.functions.prompts.user_input import OpenAIUserMessage

openai_chat = OpenAIChat(name='openai_chat',
                         llm_name='gpt-3.5-turbo',
                         temperature=0,
                         n=1)

system_role_comedian = OpenAISystemMessage(name="Role",
                                           content="You are a comedian. Given a subject, you generate a joke about it.")

comedian_chat_template = OpenAIPromptTemplate(name="Loan Classifier Template",
                                              sections=[
                                                  system_role_comedian,
                                                  OpenAIUserMessage(name="name"),
                                              ])

async def main():

    messages = await comedian_chat_template(message="Subject: waking up grumpy\n")
    response = await openai_chat(**messages)
    print(response)

if __name__ == '__main__':
    asyncio.run(main())
