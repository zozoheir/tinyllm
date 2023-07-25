from tinyllm.functions.function import Function


class OpenAISystemMessage(Function):
    def __init__(self, content, **kwargs):
        super().__init__(**kwargs)
        self.content = content

    async def run(self, **kwargs):
        return {'role': 'system',
                'content': self.content}
