import asyncio

from tinyllm.functions.prompts.openai_chat.system import OpenAISystemMessage
from tinyllm.functions.prompts.template import OpenAIPromptTemplate
from tinyllm.functions.prompts.user_input import OpenAIUserMessage


async def main():
    translator_role = OpenAISystemMessage(name="Introduction",
                                          content="You translate sentences from English to French")
    translator_prompt_template = OpenAIPromptTemplate(name="Example Template",
                                                      sections=[
                                                          translator_role,
                                                          OpenAIUserMessage(name="name"),
                                                      ])

    user_inputs = await translator_prompt_template(message="The sky is blue and the grass is green.")
    print(user_inputs)


if __name__ == "__main__":
    asyncio.run(main())
