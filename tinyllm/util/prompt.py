from typing import List

from tinyllm.util.helpers import count_tokens


class Prompt:
    def __init__(self, max_token_size):
        self.max_token_size = max_token_size
        self.messages = []

    @property
    def size(self):
        return count_tokens(self.messages)

    def add_context(self, messages: List[dict]):
        new_messages_size = count_tokens(messages)
        if self.size + new_messages_size <= self.max_token_size:
            self.messages += messages
        else:
            raise ValueError("Prompt size exceeded")

    @property
    def leftover_size(self):
        return self.max_token_size - self.size