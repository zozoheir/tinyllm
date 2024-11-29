from typing import List

from tinyllm.util.message import *


class Example:
    def __init__(self,
                 user_message: UserMessage,
                 assistant_message: AssistantMessage):
        self.user_message = user_message
        self.assistant_message = assistant_message


class ExampleManager:

    def __init__(self,
                 example_selector=None,
                 constant_examples: Union[List[Example], Example] = None):
        self.constant_examples = constant_examples if type(constant_examples) == list else ([constant_examples]) if constant_examples else []
        self.example_selector = example_selector
