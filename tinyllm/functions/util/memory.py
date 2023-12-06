import abc

from tinyllm.function import Function


class Memory(Function, abc.ABC):
    """
    This function is used as a template for all chat history functions.
    It should implement: add_message, get_history abstract methods.
    """

    def __init__(self,
                 **kwargs):
        super().__init__(**kwargs)
        self.memories = None

    async def load_memory(self, **kwargs):
        pass

    async def store_memory(self, **kwargs):
        pass
