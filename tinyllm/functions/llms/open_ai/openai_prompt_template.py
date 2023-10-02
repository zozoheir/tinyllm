from typing import List, Dict, Optional, Union

from tinyllm.functions.function import Function
from tinyllm.functions.llms.example_selector import VectorStoreExampleSelector, LocalExampleSelector
from tinyllm.functions.llms.open_ai.util.helpers import get_system_message, count_tokens, get_user_message, \
    get_assistant_message
from tinyllm.functions.validator import Validator


class InitValidator(Validator):
    system_role: str
    messages: Optional[List[Dict]]
    example_selector: Optional[Union[VectorStoreExampleSelector, LocalExampleSelector]]


class InputValidator(Validator):
    openai_msg: Dict[str, str]
    memories: List[Dict]


class OutputValidator(Validator):
    messages: List[Dict]


class OpenAIPromptTemplate(Function):

    def __init__(self,
                 system_role="You are a helpful assistant",
                 messages = [],
                 example_selector = None,
                 **kwargs):
        val = InitValidator(system_role=system_role,
                            messages=messages,
                            example_selector=example_selector,
                            **kwargs)
        super().__init__(input_validator=InputValidator,
                         output_validator=OutputValidator,
                         **kwargs)
        self.system_role = system_role
        self.messages = [get_system_message(self.system_role)] + messages
        self.example_selector = example_selector

    async def run(self,
                  **kwargs):

        messages = self.messages + kwargs['memories']

        if self.example_selector:
            best_examples = await self.example_selector(user_question=kwargs['openai_msg']['content'])
            for good_example in best_examples['best_examples']:
                messages.append(get_user_message(good_example['USER']))
                messages.append(get_assistant_message(str(good_example['ASSISTANT'])))

        messages.append(kwargs['openai_msg'])
        return {'messages': messages}

    @property
    def size(self):
        return count_tokens(self.messages,
                            header='',
                            ignore_keys=[])
