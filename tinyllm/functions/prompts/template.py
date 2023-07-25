from typing import List, Type, Dict

from tinyllm.functions.function import Function
from tinyllm.functions.validator import Validator


class OpenAIPromptTemplateInitValidator(Validator):
    sections: List


class OpenAIPromptTemplateOutputValidator(Validator):
    messages: List[Dict[str, str]]


class OpenAIPromptTemplate(Function):
    def __init__(self,
                 sections,
                 **kwargs):
        val = OpenAIPromptTemplateInitValidator(sections=sections)
        super().__init__(output_validator=OpenAIPromptTemplateOutputValidator,
                         **kwargs)
        self.sections = sections

    async def run(self, **kwargs):
        messages = []
        for section in self.sections:
            messages.append(await section(**kwargs))
        return {'messages': messages}
