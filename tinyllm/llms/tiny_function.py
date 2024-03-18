import asyncio
import functools
import json
from textwrap import dedent

from pydantic import BaseModel
from typing import Type


def model_to_code(model) -> str:
    fields = model.__fields__
    field_defs = []
    for field_name, field in fields.items():
        field_type = field.annotation
        field_defs.append(
            f"    {field_name}: {field_type}" + f" - Desc: {field.description}" if field.description else "")
    model_code = "Model:\n" + "\n".join(field_defs)
    return model_code


class MissingBlockException(Exception):
    pass


def tiny_function(output_model: Type[BaseModel], example_output: dict = 'your extracted output here'):
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Validate that input is provided as a string
            if not isinstance(kwargs.get('content', ''), str):
                return {
                    'status': 'error',
                    "message": "Input content must be a string"
                }

            # Extract system_role from the docstring of the decorated function
            model_code = model_to_code(output_model)
            system_role = func.__doc__.strip()
            system_role = system_role + "\n" + dedent(f"""
DATA MODEL
Your output must respect the following pydantic model:
{model_code}

OUTPUT FORMAT
Your output must be in JSON

EXAMPLE
Here is an example of the expected output:
{json.dumps(example_output)}

You must respect all the requirements above.
""")

            # Prepare the agent with necessary parameters
            from tinyllm.agent.agent import Agent
            from tinyllm.llms.lite_llm import LiteLLM

            # examples = [
            #    get_openai_message(role='user', content=EXAMPLE_INPUT),
            #    get_openai_message(role='assistant', content=example_output)
            # ]
            agent = Agent(
                name=func.__name__,
                system_role=system_role,
                llm=LiteLLM(),
                # example_manager=ExampleManager(constant_examples=)
            )

            loop = asyncio.get_event_loop()

            result = loop.run_until_complete(agent(content=kwargs['content'],
                                                   resopnse_format={"type": "json_object"}))
            if result['status'] == 'success':
                msg_content = result['output']['response']['choices'][0]['message']['content']
                try:
                    parsed_output = json.loads(msg_content)
                    return {
                        'status': 'success',
                        'output': output_model(**parsed_output)
                    }
                except (ValueError, json.JSONDecodeError) as e:
                    return {"message": "Parsing error", "details": str(e),
                            'status': 'error'}
            else:
                return {
                    'status': 'error',
                    "message": "Agent failed", "details": result['response']['status']}

        return wrapper

    return decorator
