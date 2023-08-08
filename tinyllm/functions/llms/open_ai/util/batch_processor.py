from typing import List, Dict

from tinyllm.functions.llms.open_ai.openai_prompt_template import OpenAIPromptTemplate
from tinyllm.functions.llms.open_ai.util.helpers import num_tokens_from_string, get_openai_api_cost, \
    openai_model_context_sizes, num_tokens_from_messages
from smartpy.utility.log_util import getLogger
from tinyllm.functions.llms.open_ai.util.job_batch import OpenAIJobBatch, LLMJobBatchSizeExceeded

logger = getLogger(__name__)

class OpenAIBatchProcessor:
    def __init__(self,
                 openai_model: str,
                 prompt_template: OpenAIPromptTemplate,
                 data_queue: List[Dict],
                 prompt_tokens_proportion: float,
                 verbose=False):
        self.openai_model = openai_model
        self.data_queue = data_queue
        self.context_size = int(openai_model_context_sizes[openai_model])
        self.prompt_template_n_tokens = int(num_tokens_from_messages(prompt_template.messages))
        self.verbose = verbose

        self.leftover_total_tokens = int(self.context_size - self.prompt_template_n_tokens)
        self.allowed_prompt_tokens = int(self.leftover_total_tokens * prompt_tokens_proportion)
        self.allowed_completion_tokens = int(self.leftover_total_tokens * (1 - prompt_tokens_proportion))
        assert int(self.allowed_prompt_tokens + self.allowed_completion_tokens)-int(self.leftover_total_tokens)<1

        logger.info("BatchProcessor initialized.")


    def get_batch_metadata(self,
                           batch):
        batch_metadata_keys = ['context_size', 'prompt_template_n_tokens', 'leftover_total_tokens',
                               'allowed_prompt_tokens', 'allowed_completion_tokens']
        batch_metadata = {key: str(value) for key, value in
                          self.__dict__.items() for key in batch_metadata_keys}
        batch_metadata['batch_n_units'] = len(batch.job_units_dicts)
        batch_metadata['batch_tokens'] = batch.token_size
        batch_metadata['batch_post_ids'] = [i['id'] for i in batch.job_units_dicts]
        batch_metadata['context_distribution'] = {
            'leftover': round(self.leftover_total_tokens / self.context_size, 2),
            'prompt_template': round(self.prompt_template_n_tokens / self.context_size, 2),
            'allowed_prompt_tokens': round(self.allowed_prompt_tokens / self.context_size, 2),
            'allowed_completion_tokens': round(
                self.allowed_completion_tokens / self.context_size, 2),
            'actual_prompt_tokens': round(batch.token_size / self.context_size, 2)
        }

        return batch_metadata


    def get_batches(self):
        generated_batches = []
        llm_job_batch = OpenAIJobBatch(token_size_limit=self.allowed_prompt_tokens,
                                       dict_header='[post]')
        for unit_dict in self.data_queue:
            try:
                llm_job_batch.add(unit_dict)
            except LLMJobBatchSizeExceeded:
                generated_batches.append(llm_job_batch)

                logger.info(f"Batch created of token size: {llm_job_batch.token_size}")
                logger.info(f"Batch created with {len(llm_job_batch.job_units_dicts)} dicts")

                llm_job_batch = OpenAIJobBatch(token_size_limit=self.allowed_prompt_tokens,
                                               dict_header='[post]')
                llm_job_batch.add(unit_dict)

        logger.info(f"{len(generated_batches)} batches generated")

        return generated_batches

    def estimate_batch_cost(self,
                            batch):
        batch_tokens = num_tokens_from_string(batch)
        batch_cost = get_openai_api_cost(model=self.openai_model,
                                         completion_tokens=self.allowed_completion_tokens,
                                         prompt_tokens=batch_tokens,
                                         total_tokens=batch_tokens + self.allowed_completion_tokens)
        per_unit_cost = batch_cost['request_cost'] / len(self.data_queue)
        return per_unit_cost
