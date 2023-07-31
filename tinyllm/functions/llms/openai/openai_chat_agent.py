import json
from datetime import datetime
from typing import List, Dict, Callable

import langfuse
from langfuse.api.model import CreateTrace, CreateSpan

from tinyllm.functions.llms.openai.openai_chat import OpenAIChat, chat_completion_with_backoff
from tinyllm.functions.llms.openai.openai_prompt_template import OpenAIPromptTemplate
from tinyllm.functions.validator import Validator
from langfuse import Langfuse


langfuse_client = Langfuse(
    public_key="pk-lf-29a89706-d49e-4795-b72e-b59e2336289a",
    secret_key="sk-lf-32250a16-2480-481a-8ea8-9fb221c65f4a",
    host="http://localhost:3030/"
)



class InitValidator(Validator):
    functions: List[Dict]
    function_callables: Dict[str, Callable]
    prompt_template: OpenAIPromptTemplate


class OpenAIChatAgent(OpenAIChat):
    def __init__(self,
                 openai_functions,
                 function_callables,
                 prompt_template,
                 **kwargs):
        val = InitValidator(functions=openai_functions,
                            function_callables=function_callables,
                            prompt_template=prompt_template)
        super().__init__(prompt_template=prompt_template,
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
        message = kwargs.pop('message')
        role = kwargs.pop('role', 'user')
        prompt = await self.generate_prompt(message=message)
        await self.memory(role=role, message=message)

        api_result = await chat_completion_with_backoff(
            model=self.llm_name,
            temperature=self.temperature,
            n=self.n,
            messages=prompt['prompt'],
            functions=self.functions,
            function_call='auto'
        )

        # if api_result['choices'][0]['message']['function_call']
        return {'response': api_result}

    async def run_openai_function(self,
                                  **kwargs):
        retrievalStartTime = datetime.datetime.now()
        function_call = kwargs['response']['choices'][0]['message']['function_call']
        callable = self.function_callables[function_call['name']]
        function_result = callable(**json.loads(function_call['arguments']))
        if self.verbose is True:
            self.span = self.trace.span(
                CreateSpan(
                    name=f"function-{function_call['name']}",
                    startTime=retrievalStartTime,
                    endTime=datetime.datetime.now(),
                    input={kwargs},
                    output=function_result
                )
            )
        return function_result

    async def process_output(self, **kwargs):
        if kwargs['response']['choices'][0]['finish_reason'] == 'function_call':
            result = await self.run_openai_function(**kwargs)
            await self.memory(role='function', message=result)
            result = await self(role='function',
                                message=result)
            return {'response': result}
        else:
            model_response = ['message']['content']
            await self.memory(role='assistant', message=model_response)
            return {'response': model_response}
