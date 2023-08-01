import json
from datetime import datetime
from typing import List, Dict, Callable

from langfuse.api.model import CreateTrace, CreateSpan, UpdateSpan

from tinyllm.functions.llms.openai.helpers import get_function_message, get_assistant_message, get_user_message
from tinyllm.functions.llms.openai.openai_chat import OpenAIChat, chat_completion_with_backoff
from tinyllm.functions.llms.openai.openai_prompt_template import OpenAIPromptTemplate
from tinyllm.functions.validator import Validator
from langfuse import Langfuse
from langfuse.api.model import CreateGeneration, Usage

langfuse_client = Langfuse(
    public_key="pk-lf-29a89706-d49e-4795-b72e-b59e2336289a",
    secret_key="sk-lf-32250a16-2480-481a-8ea8-9fb221c65f4a",
    host="http://localhost:3030/"
)


class OpenAIChatAgentInitValidator(Validator):
    functions: List[Dict]
    function_callables: Dict[str, Callable]
    prompt_template: OpenAIPromptTemplate


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
        super().__init__(prompt_template=prompt_template,
                         **kwargs)
        self.functions = openai_functions
        self.prompt_template = prompt_template
        self.function_callables = function_callables

        self.trace = langfuse_client.trace(CreateTrace(
            name=self.name,
            userId="test",
            metadata={
                "model": self.llm_name,
                "modelParameters": self.parameters,
                "prompt": self.prompt_template.messages,
                "functions": self.functions,
                "functionCallables": self.function_callables,
            }
        ))

    @property
    def parameters(self):
        return

    async def run(self, **kwargs):
        start_time = datetime.now()

        user_msg = get_user_message(message=kwargs['message'])
        messages = await self.prompt_template(openai_msg=user_msg,
                                              memories=self.memory.memories)
        await self.add_memory(new_memory=user_msg)
        api_result = await chat_completion_with_backoff(
            model=self.llm_name,
            temperature=self.temperature,
            n=self.n,
            messages=messages['messages'],
            functions=self.functions,
            function_call='auto',
            max_tokens=self.max_tokens,
        )
        if api_result['choices'][0]['finish_reason'] == 'function_call':
            self.trace.generation(CreateGeneration(
                name=f"Calling function: {api_result['choices'][0]['message']['function_call']['name']}",
                startTime=start_time,
                endTime=datetime.now(),
                model=self.llm_name,
                modelParameters=self.parameters,
                prompt=messages['messages'],
                metadata=api_result['choices'][0],
                usage=Usage(promptTokens=50, completionTokens=49),
            ))
        else:
            assistant_response = api_result['choices'][0]['message']['content']
            self.trace.generation(CreateGeneration(
                name=f"Assistant response",
                startTime=start_time,
                endTime=datetime.now(),
                model=self.llm_name,
                modelParameters=self.parameters,
                prompt=messages['messages'],
                completion=assistant_response,
                metadata=api_result['choices'][0],
                usage=Usage(promptTokens=50, completionTokens=49),
            ))
        return {'response': api_result}


    async def process_output(self, **kwargs):
        if kwargs['response']['choices'][0]['finish_reason'] == 'function_call':
            function_name = kwargs['response']['choices'][0]['message']['function_call']['name']
            # Call the function
            function_result = await self.run_openai_function(
                function_call_info=kwargs['response']['choices'][0]['message']['function_call'])

            new_memory = get_function_message(content=function_result,
                                              name=function_name)

            # Now create prompt with function result included
            messages = await self.prompt_template(openai_msg=new_memory,
                                                  memories=self.memory.memories)
            # Save to memory
            await self.add_memory(new_memory=new_memory)

            # Get final assistant response with function call result by removing available functions
            start_time = datetime.now()
            api_result = await chat_completion_with_backoff(
                model=self.llm_name,
                temperature=self.temperature,
                n=self.n,
                messages=messages['messages'],
            )
            assistant_response = api_result['choices'][0]['message']['content']
            self.trace.generation(CreateGeneration(
                name=f"End: Agent response",
                startTime=start_time,
                endTime=datetime.now(),
                model=self.llm_name,
                modelParameters=self.parameters,
                prompt=messages['messages'],
                completion=assistant_response,
                metadata=api_result['choices'][0],
                usage=Usage(promptTokens=50, completionTokens=49),
            ))


        else:
            assistant_response = kwargs['response']['choices'][0]['message']['content']
            new_memory = get_assistant_message(content=assistant_response)
            await self.add_memory(new_memory=new_memory)

        return {'response': assistant_response}

    async def run_openai_function(self,
                                  function_call_info):
        start_time = datetime.now()
        callable = self.function_callables[function_call_info['name']]
        function_args = json.loads(function_call_info['arguments'])

        span = self.trace.span(
            CreateSpan(
                name=f"Running function : {function_call_info['name']}",
                startTime=start_time,
                input=function_args,
            )
        )

        function_result = callable(**function_args)

        span.update(UpdateSpan(endTime=datetime.now(), output={'output': function_result}))
        return function_result
