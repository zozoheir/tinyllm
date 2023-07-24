import asyncio

import openai

from tinyllm.functions.llms.openai_chat import OpenAIChat

model_name = 'gpt-3.5-turbo'
messages = [
    {
        "role": "system",
        "content": "Summarize content you are provided with for a second-grade student."
    },
    {
        "role": "user",
        "content": "Jupiter is the fifth planet from the Sun and the largest in the Solar System. It is a gas giant with a mass one-thousandth that of the Sun, but two-and-a-half times that of all the other planets in the Solar System combined. Jupiter is one of the brightest objects visible to the naked eye in the night sky, and has been known to ancient civilizations since before recorded history. It is named after the Roman god Jupiter.[19] When viewed from Earth, Jupiter can be bright enough for its reflected light to cast visible shadows,[20] and is on average the third-brightest natural object in the night sky after the Moon and Venus."
    }
]

openai.api_key = "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"
openai_chat = OpenAIChat(name='openai_chat',
                         model_name='gpt-3.5-turbo',
                         temperature=0,
                         n=1)

# Call the function with the provided messages
response = asyncio.run(openai_chat(messages=messages))

# Print the response
print(response)
