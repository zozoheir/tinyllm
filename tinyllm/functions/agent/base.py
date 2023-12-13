from tinyllm.functions.util.helpers import get_openai_message


class AgentBase:

    async def memorize(self,
                       message):
        if self.memory is not None:
            return await self.memory(message=message)

    async def prepare_messages(self,
                               message):
        # system prompt
        # memories
        # constant examples
        # selected examples
        # input message
        system_role = get_openai_message(role='system',
                                         content=self.manager_llm.system_role)
        examples = []

        examples += self.example_manager.constant_examples
        if self.example_manager.example_selector.example_dicts and message['role'] == 'user':
            best_examples = await self.example_manager.example_selector(input=message['content'])
            for good_example in best_examples['output']['best_examples']:
                usr_msg = get_openai_message(role='user', content=good_example['user'])
                assistant_msg = get_openai_message(role='assistant', content=good_example['assistant'])
                examples.append(usr_msg)
                examples.append(assistant_msg)

        memories = [] if self.memory is None else self.memory.get_memories()

        messages = [system_role] \
                   + memories + \
                   examples + \
                   [message]
        return messages

