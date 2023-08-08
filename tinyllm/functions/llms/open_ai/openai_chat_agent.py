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
        self.functions = openai_functions
        self.prompt_template = prompt_template
        self.function_callables = function_callables

    async def run(self, **kwargs):
        kwargs, call_metadata, messages = await self.process_input(openai_message=get_user_message(kwargs['message']),
                                                                   **kwargs)
        api_result = await self.get_completion(
            model=self.llm_name,
            temperature=self.temperature,
            n=self.n,
            messages=messages['messages'],
            functions=self.functions,
            function_call='auto',
            max_tokens=self.max_tokens,
        )
        parameters = self.parameters
        parameters['cost_summary'] = get_openai_api_cost(model=self.llm_name,
                                                         completion_tokens=api_result["usage"]['completion_tokens'],
                                                         prompt_tokens=api_result["usage"]['prompt_tokens'])
        parameters['total_cost'] = self.total_cost
        if api_result['choices'][0]['finish_reason'] == 'function_call':
            self.llm_trace.create_span(
                name=f"Calling function: {api_result['choices'][0]['message']['function_call']['name']}",
                startTime=datetime.now(),
                metadata=api_result['choices'][0],
            )
        else:
            assistant_response = api_result['choices'][0]['message']['content']

        return {'response': api_result}

    async def process_output(self, **kwargs):

        # Case if function call
        if kwargs['response']['choices'][0]['finish_reason'] == 'function_call':
            # Call the function
            function_name = kwargs['response']['choices'][0]['message']['function_call']['name']
            function_result = await self.run_agent_function(
                function_call_info=kwargs['response']['choices'][0]['message']['function_call'])

            # Append function result to memory
            function_msg = get_function_message(
                content=function_result,
                name=function_name
            )

            kwargs, call_metadata, messages = await self.process_input(
                openai_message=get_function_message(name=function_name,
                                                    content=function_msg['content']),
                **kwargs
            )

            # Get final assistant response with function call result by removing available functions
            api_result = await self.get_completion(
                model=self.llm_name,
                temperature=self.temperature,
                n=self.n,
                messages=messages['messages'],
            )
            assistant_response = api_result['choices'][0]['message']['content']

        else:
            assistant_response = kwargs['response']['choices'][0]['message']['content']
            function_msg = get_assistant_message(content=assistant_response)
            await self.add_memory(new_memory=function_msg)

        return {'response': assistant_response}

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
        self.llm_trace.update_span(endTime=datetime.now(), output={'output': function_result})
        return function_result
