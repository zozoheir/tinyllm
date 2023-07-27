from tinyllm.functions.llms.memory import Memory


class OpenAIMemory(Memory):
    def __init__(self,
                 **kwargs):
        super().__init__(
            **kwargs
        )
        self.memories = []

    async def run(self, **kwargs):
        self.memories.append({'role': kwargs['role'],'content': kwargs['message']})
        return {'success': True}
