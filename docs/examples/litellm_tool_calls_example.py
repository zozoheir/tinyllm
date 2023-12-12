from pprint import pprint

from litellm import completion

from tinyllm.functions.helpers import get_openai_message

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location"],
            },
        }
    }
]
messages = [{
    "role": "user",
    "content": "What's the weather like in Boston today?"
}]
tool_call_completion = completion(
    model="gpt-3.5-turbo",
    messages=messages,
    tools=tools,
)
assistant_message = tool_call_completion.model_dump()["choices"][0]["message"]
messages.append(assistant_message)
messages.append(get_openai_message(
    name='get_current_weather',
    role='tool',
    content='26 degrees celsius',
    tool_call_id=assistant_message['tool_calls'][0]['id']
))

user_response_completion = completion(
    model="gpt-3.5-turbo",
    messages=messages,
    tools=tools,
)

pprint(messages)
