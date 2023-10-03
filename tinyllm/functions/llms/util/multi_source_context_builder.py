from typing import List, Dict, Optional
from tinyllm.functions.llms.open_ai.util.helpers import count_tokens, get_user_message
from tinyllm.functions.llms.util.context_builder import ContextBuilder
from tinyllm.util.prompt_util import stringify_dict_list


class MultiSourceDocsContextBuilder(ContextBuilder):
    def __init__(self, start_string: str, end_string: str, available_token_size: int):
        self.start_string = start_string
        self.end_string = end_string
        self.available_token_size = available_token_size
        self.fitted_input = None
        self.context = None

    def get_context(self,
                    docs: List,
                    weights: Optional[List[float]] = None,
                    output_format="str",
                    header="[doc]",
                    ignore_keys=[],
                    ):
        count_tokens(docs[0])
        # If list of strings, convert to list of dicts
        final_docs = []
        for source_docs in docs:
            if isinstance(source_docs, List) and isinstance(source_docs[0], str):
                source_docs = [{"content": doc} for doc in source_docs]
                final_docs.append(source_docs)
            else:
                final_docs.append(source_docs)

        # Fit the multiple sources of docs based on weights
        fitted_list = self.fit(final_docs,
                               weights,
                               header=header,
                               ignore_keys=ignore_keys)

        # Convert to appropriate format
        fitted_context_string = stringify_dict_list(header=header, dicts=fitted_list, ignore_keys=ignore_keys)

        # Format the final context
        formatted_context = self.format_context(fitted_context_string)

        if output_format == "openai":
            return get_user_message(formatted_context)
        elif output_format == "str":
            return formatted_context


    def fit(self,
            docs: List,
            weights: Optional[List[float]] = None,
            header="[doc]",
            ignore_keys=[],
            ) -> List[Dict]:
        # If weights are not provided, distribute docs evenly
        if not weights:
            weights = [1 / len(docs)] * len(docs)

        if len(weights) != len(docs):
            raise ValueError("Length of weights must be equal to the number of doc sources.")

        # Normalize weights to ensure they sum up to 1
        total_weight = sum(weights)
        weights = [weight / total_weight for weight in weights]

        # Calculate token size for each source based on weights
        token_sizes = [int(weight * self.available_token_size) for weight in weights]

        fitted_docs = []

        for doc_list, token_size in zip(docs, token_sizes):
            # For each doc source, count how many docs can fit into its token size
            current_tokens = 0
            for doc in doc_list:
                doc_tokens = count_tokens(doc,
                                          header=header,
                                          ignore_keys=ignore_keys)
                if current_tokens + doc_tokens <= token_size:
                    fitted_docs.append(doc)
                    current_tokens += doc_tokens
                else:
                    break

        return fitted_docs
