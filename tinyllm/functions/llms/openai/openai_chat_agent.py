import json
from datetime import datetime
from typing import List, Dict, Callable, Optional

import langfuse
from langfuse.api.model import CreateTrace, CreateSpan

from tinyllm.functions.llms.openai.helpers import get_function_message, get_assistant_message, get_user_message
from tinyllm.functions.llms.openai.openai_chat import OpenAIChat, chat_completion_with_backoff
from tinyllm.functions.llms.openai.openai_prompt_template import OpenAIPromptTemplate
from tinyllm.functions.validator import Validator
from langfuse import Langfuse

langfuse_client = Langfuse(
    public_key="pk-lf-29a89706-d49e-4795-b72e-b59e2336289a",
    secret_key="sk-lf-32250a16-2480-481a-8ea8-9fb221c65f4a",
    host="http://localhost:3030/"
)


class OpenAIChatAgentInitValidator(Validator):
    functions: List[Dict]
    function_callables: Dict[str, Callable]
    prompt_template: OpenAIPromptTemplate


class OpenAIChatAgentInputValidator(Validator):
    user_content: str


class OpenAIChatAgentOutputValidator(Validator):
    response: Dict


class OpenAIChatAgent(OpenAIChat):
    def __init__(self,
                 openai_functions,
                 function_callables,
                 prompt_template,
                 **kwargs):
        val = OpenAIChatAgentInitValidator(functions=openai_functions,
                                           function_callables=function_callables,
                                           prompt_template=prompt_template)
        super().__init__(input_validator=OpenAIChatAgentInputValidator,
                         prompt_template=prompt_template,
                         **kwargs)
        self.functions = openai_functions
        self.prompt_template = prompt_template
        self.function_callables = function_callables
        if self.verbose is True:
            self.trace = langfuse_client.trace(CreateTrace(
                name=self.name,
                userId="test",
                metadata={
                    "function_id": self.function_id,
                }
            ))

    async def run(self, **kwargs):
        user_msg = get_user_message(message=kwargs['user_content'])
        messages = await self.prompt_template(openai_msg=user_msg,
                                              memories=self.memory.memories)
        await self.add_memory(new_memory=user_msg)
        api_result = await chat_completion_with_backoff(
            model=self.llm_name,
            temperature=self.temperature,
            n=self.n,
            messages=messages['messages'],
            functions=self.functions,
            function_call='auto'
        )
        return {'response': api_result}

    async def add_memory(self, new_memory):
        await self.memory(openai_message=new_memory)

    async def process_output(self, **kwargs):
        if kwargs['response']['choices'][0]['finish_reason'] == 'function_call':

            # Call the function
            function_name = kwargs['response']['choices'][0]['message']['function_call']['name']
            function_result = await self.run_openai_function(
                function_call_info=kwargs['response']['choices'][0]['message']['function_call'])

            new_memory = get_function_message(content=function_result,
                                              name=function_name)

            # Now create prompt with function result included
            messages = await self.prompt_template(openai_msg=new_memory,
                                                  memories=self.memory.memories)
            # Save to memory
            await self.add_memory(new_memory=new_memory)

            api_result = await chat_completion_with_backoff(
                model=self.llm_name,
                temperature=self.temperature,
                n=self.n,
                messages=messages['messages'],
            )
            return {'response': api_result}

        else:
            model_response = kwargs['response']['choices'][0]['message']['content']
            new_memory = get_assistant_message(content=model_response)
            await self.add_memory(new_memory=new_memory)

    async def run_openai_function(self,
                                  function_call_info):
        retrievalStartTime = datetime.now()
        callable = self.function_callables[function_call_info['name']]
        function_args = json.loads(function_call_info['arguments'])
        function_result = callable(**function_args)
        return function_result
