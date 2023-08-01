from typing import List, Dict

from tinyllm.functions.function import Function
from tinyllm.functions.llms.openai.helpers import get_user_message, get_system_message
from tinyllm.functions.validator import Validator

class InitValidator(Validator):
    system_role: str


class InputValidator(Validator):
    openai_msg: Dict[str, str]
    memories: List[Dict]

class OutputValidator(Validator):
    messages: List[Dict]


class OpenAIPromptTemplate(Function):

    def __init__(self,
                 system_role="You are a helpful assistant",
                 messages=[],
                 **kwargs):
        val = InitValidator(system_role=system_role)
        super().__init__(**kwargs,
                         input_validator=InputValidator,
                         output_validator=OutputValidator)
        self.system_role = system_role
        self.messages = [get_system_message(self.system_role)] + [get_user_message(msg) for msg in messages]


    async def run(self,
                  **kwargs):
        messages =  self.messages + kwargs['memories'] + [kwargs['openai_msg']]
        return {'messages': messages}