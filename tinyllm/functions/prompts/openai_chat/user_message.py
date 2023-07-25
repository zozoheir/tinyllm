from tinyllm.functions.function import Function


class UserMessage(Function):
    def __init__(self,
                 **kwargs):
        super().__init__(name='user_message',
                         **kwargs)
        self.content = None

    async def run(self, **kwargs):
        return {'content': kwargs['content']}

