import asyncio
import functools
import json
from textwrap import dedent

from pydantic import BaseModel, create_model
from typing import Type, Dict, Any, Union, List

from tinyllm.util.message import Content
from tinyllm.validator import Validator


def model_to_string(model) -> str:
    fields = model.__fields__
    field_defs = []
    for field_name, field in fields.items():
        field_type = field.annotation.__name__
        description = getattr(field, 'description', None)
        description = f" | Description: {field.description}" if description else ""
        field_defs.append(
            f"    {field_name}: {field_type}" + description)
    model_prompt = "Model:\n" + "\n".join(field_defs) if field_defs else ""
    return model_prompt


class MissingBlockException(Exception):
    pass


def create_pydantic_model_from_dict(data: Dict[str, Any]) -> BaseModel:
    fields = {key: (type(value), ...) for key, value in data.items()}
    JSONOutput = create_model('JSONOutput', **fields)
    model_instance = JSONOutput(**data)

    return model_instance


def model_to_string(model) -> str:
    fields = model.__fields__
    field_defs = []
    for field_name, field in fields.items():
        field_type = field.annotation.__name__
        description = getattr(field, 'description', None)
        description = f" | Description: {field.description}" if description else ""
        field_defs.append(
            f"    {field_name}: {field_type}" + description)
    model_prompt = "Model:\n" + "\n".join(field_defs) if field_defs else ""
    return model_prompt


def get_function_prompt(func,
                        example_output,
                        output_model) -> str:
    system_prompt = func.__doc__.strip() + '\n\n' + dedent("""
    OUTPUT FORMAT
    Your output must be in JSON

    {data_model}

    {example}

    You must respect all the requirements above.
    """)

    example_output_prompt = dedent(f"""
    EXAMPLE
    Here is an example of the expected output:
    {json.dumps(example_output)}
    """) if example_output else ""

    pydantic_model = model_to_string(output_model) if output_model else None

    data_model_prompt = dedent(f"""
    DATA MODEL
    Your output must respect the following pydantic model:
    {pydantic_model}
    """) if pydantic_model else ""

    final_prompt = system_prompt.format(pydantic_model=pydantic_model,
                                        example=example_output_prompt,
                                        data_model=data_model_prompt)
    return final_prompt


class TinyFunctionInputValidator(Validator):
    content: Union[str, List[Content]]


def tiny_function(output_model: Type[BaseModel] = None,
                  example_output=None):
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            TinyFunctionInputValidator(**kwargs)

            system_role = get_function_prompt(func=func,
                                              output_model=output_model,
                                              example_output=example_output)

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
            result = await agent(content=kwargs['content'],
                                 response_format={"type": "json_object"})
            if result['status'] == 'success':
                msg_content = result['output']['response']['choices'][0]['message']['content']
                try:
                    parsed_output = json.loads(msg_content)
                    if output_model is None:
                        function_output_model = create_pydantic_model_from_dict(parsed_output)
                    else:
                        function_output_model = output_model(**parsed_output)
                    return {
                        'status': 'success',
                        'output': function_output_model
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
