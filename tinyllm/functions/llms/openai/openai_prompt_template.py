from typing import List, Dict
from tinyllm.functions.llms.openai.helpers import get_user_message, get_system_message, get_function_message
from tinyllm.functions.llms.prompt_template import PromptTemplate
from tinyllm.functions.validator import Validator
from tinyllm.util import prompt_util
from tinyllm.util.prompt_util import shuffle_with_freeze


class InitValidator(Validator):
    system_role: str


class InputValidator(Validator):
    openai_msg: Dict[str, str]
    memories: List[Dict]

class OutputValidator(Validator):
    messages: List[Dict]


class OpenAIPromptTemplate(PromptTemplate):

    def __init__(self,
                 system_role="You are a helpful assistant",
                 messages=[],
                 **kwargs):
        val = InitValidator(system_role=system_role)
        super().__init__(**kwargs,
                         input_validator=InputValidator,
                         messages=messages,
                         output_validator=OutputValidator)
        self.system_role = system_role
        self.messages = messages

    async def run(self,
                  **kwargs):
        messages = [get_system_message(self.system_role)] + [get_user_message(section) for section in self.messages] + kwargs['memories'] + [kwargs['openai_msg']]
        return {'messages': messages}