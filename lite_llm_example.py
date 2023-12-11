import litellm
import openai
from litellm import acompletion
import os

openai.api_key = os.environ['OPENAI_API_KEY']
litellm.set_verbose = True

messages = [
    {'role': 'system',
     'content': "You are a helpful agent that can answer questions using available tools."},
    {'role': 'user', 'content': "What is the weather in San Francisco, CA? today"},
]

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "format": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "The temperature unit to use. Infer this from the users location.",
                    },
                },
                "required": ["location", "format"],
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_n_day_weather_forecast",
            "description": "Get an N-day weather forecast",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "format": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "The temperature unit to use. Infer this from the users location.",
                    },
                    "num_days": {
                        "type": "integer",
                        "description": "The number of days to forecast",
                    }
                },
                "required": ["location", "format", "num_days"]
            },
        }
    },
]


async def main():
    response = await acompletion(
        model='gpt-3.5-turbo',
        temperature=0,
        n=1,
        max_tokens=100,
        messages=messages,
        stream=True,
        tools=tools,
    )
    chunks = []
    async for chunk in response:
        chunks.append(chunk)
    return chunks

if __name__ == '__main__':
    import asyncio
    result = asyncio.run(main())
    print(result)
