from tinyllm.functions.function import Function


class OpenAIUserMessage(Function):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)

    async def run(self, **kwargs):
        return {'role': 'user',
                'content': kwargs['message']}
