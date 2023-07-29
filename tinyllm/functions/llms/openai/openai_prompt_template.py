from typing import List, Dict
from tinyllm.functions.llms.openai.helpers import get_user_message, get_system_message
from tinyllm.functions.llms.prompt_template import PromptTemplate
from tinyllm.functions.validator import Validator
from tinyllm.util import prompt_util
from tinyllm.util.prompt_util import shuffle_with_freeze


class InitValidator(Validator):
    system_role: str


class OutputValidator(Validator):
    prompt: List[Dict[str, str]]


class OpenAIPromptTemplate(PromptTemplate):

    def __init__(self,
                 system_role="You are a helpful assistant",
                 messages=[],
                 **kwargs):
        val = InitValidator(system_role=system_role)
        super().__init__(**kwargs,
                         messages=messages,
                         output_validator=OutputValidator)
        self.system_role = system_role
        self.messages = messages

    async def generate_prompt(self,
                              method='multi',
                              shuffle=False,
                              freeze=[],
                              **kwargs) -> List[str]:
        message = kwargs['message']
        if shuffle is True:
            messages = shuffle_with_freeze(self.messages, freeze)
        else:
            messages = self.messages

        if method == 'multi':
            messages = [get_user_message(section) for section in messages] + [get_user_message(message)]
        elif method == 'single':
            messages = [get_user_message(prompt_util.concatenate_strings(messages + [get_user_message(message)]))]

        if self.system_role:
            return [get_system_message(self.system_role)]+messages
        else:
            return messages

        return
