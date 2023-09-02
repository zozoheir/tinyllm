import json
from datetime import datetime
from typing import List, Dict, Callable

from tinyllm.functions.llms.open_ai.util.helpers import get_function_message, get_assistant_message, get_user_message, \
    get_openai_api_cost
from tinyllm.functions.llms.open_ai.openai_chat import OpenAIChat
from tinyllm.functions.llms.open_ai.openai_prompt_template import OpenAIPromptTemplate
from tinyllm.functions.validator import Validator


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
                 prompt_template=OpenAIPromptTemplate(name="standard_prompt_template",
                                                      is_traced=False),
                 **kwargs):
        val = OpenAIChatAgentInitValidator(functions=openai_functions,
                                           function_callables=function_callables,
                                           prompt_template=prompt_template)
        super().__init__(prompt_template=prompt_template,
                         **kwargs)
        self.openai_functions = openai_functions
        self.prompt_template = prompt_template
        self.function_callables = function_callables

    async def run(self, **kwargs):
        message = kwargs.pop('message')
        llm_name = kwargs['llm_name'] if kwargs['llm_name'] is not None else self.llm_name
        temperature = kwargs['temperature'] if kwargs['temperature'] is not None else self.temperature
        max_tokens = kwargs['max_tokens'] if kwargs['max_tokens'] is not None else self.max_tokens
        call_metadata = kwargs['call_metadata'] if kwargs['call_metadata'] is not None else {}

        messages = await self.process_input_message(openai_message=get_user_message(message))

        api_result = await self.get_completion(
            messages=messages['messages'],
            llm_name=llm_name,
            temperature=temperature,
            max_tokens=max_tokens,
            n=self.n,
            call_metadata=call_metadata,
            functions=self.openai_functions,
            function_call='auto',
        )

        call_metadata['cost_summary'] = get_openai_api_cost(model=self.llm_name,
                                                         completion_tokens=api_result["usage"]['completion_tokens'],
                                                         prompt_tokens=api_result["usage"]['prompt_tokens'])
        call_metadata['total_cost'] = self.total_cost
        if api_result['choices'][0]['finish_reason'] == 'function_call':
            self.llm_trace.create_span(
                name=f"Calling function: {api_result['choices'][0]['message']['function_call']['name']}",
                startTime=datetime.now(),
                metadata={'api_result':api_result['choices'][0]},
            )
        return {'response': api_result}




    async def process_output(self, **kwargs):

        # Case if OpenAI decides function call
        if kwargs['response']['choices'][0]['finish_reason'] == 'function_call':
            # Call the function
            function_name = kwargs['response']['choices'][0]['message']['function_call']['name']
            function_result = await self.run_agent_function(
                function_call_info=kwargs['response']['choices'][0]['message']['function_call']
            )

            # Append function result to memory
            function_msg = get_function_message(
                content=function_result,
                name=function_name
            )

            # Generate input messages with the function call content
            messages = await self.process_input_message(
                openai_message=get_function_message(name=function_name,
                                                    content=function_msg['content']),
                **kwargs
            )

            # Make API call with the function call content
            assistant_response = await self.get_assistant_response_with_function_result(
                llm_name=self.llm_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                n=self.n,
                messages=messages['messages'],
            )

        else:
            # If no function call, just return the result
            assistant_response = kwargs['response']['choices'][0]['message']['content']
            function_msg = get_assistant_message(content=assistant_response)
            await self.add_memory(new_memory=function_msg)

        return {'response': assistant_response}

    async def get_assistant_response_with_function_result(self,
                                                          llm_name,
                                                          temperature,
                                                          max_tokens,
                                                          n,
                                                          messages):
        # Remove functions arg to get final assistant response
        api_result = await self.get_completion(
            messages=messages,
            llm_name=llm_name,
            temperature=temperature,
            max_tokens=max_tokens,
            n=n,
        )
        assistant_response = api_result['choices'][0]['message']['content']
        return assistant_response


    async def run_agent_function(self,
                                 function_call_info):
        start_time = datetime.now()
        callable = self.function_callables[function_call_info['name']]
        function_args = json.loads(function_call_info['arguments'])

        self.llm_trace.update_span(
            name=f"Running function : {function_call_info['name']}",
            startTime=start_time,
            input=function_args,
        )

        function_result = callable(**function_args)
        self.llm_trace.update_span(endTime=datetime.now(), output={'output': str(function_result)})
        return function_result
