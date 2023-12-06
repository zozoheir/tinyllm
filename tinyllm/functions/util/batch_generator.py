from typing import List, Dict

from smartpy.utility.log_util import getLogger
from tinyllm.functions.llms.openai.openai_prompt_template import OpenAIPromptTemplate
from tinyllm.functions.util.helpers import OPENAI_MODELS_CONTEXT_SIZES, count_tokens, \
    count_openai_messages_tokens

logger = getLogger(__name__)

class OpenAIBatchGenerator:

    def __init__(self,
                 openai_model,
                 dicts_list: List[Dict],
                 prompt_template: OpenAIPromptTemplate,
                 expected_completion_to_input_multiplier: float):
        # [---prompt_template---][---batch_input---][---completion_input---]
        self.dicts_list = dicts_list
        self.prompt_template = prompt_template
        self.completion_to_input_multiplier = expected_completion_to_input_multiplier
        self.model_context_size = int(OPENAI_MODELS_CONTEXT_SIZES[openai_model])

        self.prompt_template_n_tokens = count_openai_messages_tokens(self.prompt_template.messages)
        self.leftover_tokens_for_completion = 0

    def generate_batches(self,
                         header,
                         ignore_keys):
        batch = []
        current_batch_size = 0
        for unit_dict in self.dicts_list:
            new_unit_dict_size = count_tokens(unit_dict,
                                               header=header,
                                               ignore_keys=ignore_keys)
            current_batch_size += new_unit_dict_size

            expected_total_tokens = current_batch_size+int(current_batch_size * self.completion_to_input_multiplier) + self.prompt_template_n_tokens
            self.leftover_tokens_for_completion = self.model_context_size - current_batch_size - self.prompt_template_n_tokens
            if expected_total_tokens < self.model_context_size :
                batch.append(unit_dict)
            else:
                logger.info(f"Yielding new batch of {len(batch)}")
                yield batch
                current_batch_size = 0
                batch = [unit_dict]

        if len(batch) > 0:
            logger.info(f"Yielding new batch of {len(batch)}")

            yield batch
