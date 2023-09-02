from typing import List, Dict

from tinyllm.functions.llms.open_ai.openai_prompt_template import OpenAIPromptTemplate
from tinyllm.functions.llms.open_ai.util.helpers import OPENAI_MODELS_CONTEXT_SIZES, count_tokens, \
    count_openai_messages_tokens


class OpenAIBatchGenerator:

    def __init__(self,
                 openai_model,
                 dicts_list: List[Dict],
                 prompt_template: OpenAIPromptTemplate,
                 expected_completion_to_input_multiplier: int,
                 max_posts_by_batch: int ):
        # [---prompt_template---][---batch_input---][---completion_input---]
        self.dicts_list = dicts_list
        self.prompt_template = prompt_template
        self.completion_to_input_multiplier = expected_completion_to_input_multiplier
        self.model_context_size = int(OPENAI_MODELS_CONTEXT_SIZES[openai_model])
        self.max_posts_by_batch = max_posts_by_batch

        self.prompt_template_n_tokens = count_openai_messages_tokens(self.prompt_template.messages)
        self.input_and_completion_leftover_size = int(self.model_context_size - self.prompt_template_n_tokens)

        # Calculate allowed tokens by splitting the leftover space between the batch input and the allowed completion
        self.optimal_batch_token_size = int(self.input_and_completion_leftover_size / (1 + self.completion_to_input_multiplier))
        self.optimal_completion_input_size = int(self.optimal_batch_token_size * self.completion_to_input_multiplier * 0.95) # 5% buffer


    def generate_batches(self):
        batch = []
        for news_dict in self.dicts_list:
            news_dict['content'] = news_dict['content'][:int(self.optimal_batch_token_size * 0.95)]
            current_batch_size = count_tokens(batch,
                                              header='[post]',
                                              ignore_keys=['timestamp', 'author', 'source','suggested_question','summary'])

            news_dict_token_size = count_tokens(news_dict,
                                                header='[post]',
                                                ignore_keys=['timestamp','author','source''suggested_question','summary'])

            if news_dict_token_size + current_batch_size < self.optimal_batch_token_size and len(batch) < self.max_posts_by_batch:
                batch.append(news_dict)
            else:
                yield batch
                batch = [news_dict]

        if len(batch) > 0:
            yield batch
