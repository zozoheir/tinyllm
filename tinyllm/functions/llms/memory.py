import abc

from tinyllm.functions.function import Function
from tinyllm.functions.validator import Validator

class InputValidator(Validator):
    message: str
    role: str


class Memory(Function, abc.ABC):
    """
    This function is used as a template for all chat history functions.
    It should implement: add_message, get_history abstract methods.
    """

    def __init__(self,
                 **kwargs):
        super().__init__(input_validator=InputValidator,
                         **kwargs)
        self.message_history = None

    async def load_memory(self, **kwargs):
        pass

    async def store_memory(self, **kwargs):
        pass
