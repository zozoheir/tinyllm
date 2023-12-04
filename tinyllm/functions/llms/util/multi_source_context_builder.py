from typing import List, Dict, Optional

from tinyllm.functions.llms.lite_llm.util.helpers import get_user_message, count_tokens
from tinyllm.functions.llms.util.context_builder import ContextBuilder
from tinyllm.util.prompt_util import stringify_dict_list


def remove_duplicate_dicts(list_of_lists):
    # Flatten the list of lists to easily identify duplicates across all lists
    flattened_list = [item for sublist in list_of_lists for item in sublist]
    # Remove duplicates while preserving order
    seen = set()
    unique_flattened_list = []
    for item in flattened_list:
        # Dictionaries are not hashable, so we use their string representation to keep track of duplicates
        item_str = str(item)
        if item_str not in seen:
            seen.add(item_str)
            unique_flattened_list.append(item)

    # Reconstruct the list of lists with unique dictionaries
    unique_list_of_lists = []
    for original_list in list_of_lists:
        new_list = []
        for item in original_list:
            if item in unique_flattened_list:
                new_list.append(item)
                # Remove the item from the unique list so it won't appear again
                unique_flattened_list.remove(item)
        unique_list_of_lists.append(new_list)

    return unique_list_of_lists


class MultiSourceDocsContextBuilder(ContextBuilder):
    def __init__(self, start_string: str, end_string: str, available_token_size: int):
        self.start_string = start_string
        self.end_string = end_string
        self.available_token_size = available_token_size
        self.fitted_list = None
        self.context = None

    def get_context(self,
                    docs: List,
                    weights: Optional[List[float]] = None,
                    output_format="str",
                    header="[doc]",
                    ignore_keys=[],
                    ):
        # If list of strings, convert to list of dicts
        final_docs = []
        for source_docs in docs:
            if isinstance(source_docs, List) and isinstance(source_docs[0], str):
                source_docs = [{"text": doc} for doc in source_docs]
                final_docs.append(source_docs)
            else:
                final_docs.append(source_docs)

        # Remove duplicates
        final_docs = remove_duplicate_dicts(final_docs)

        # Fit the multiple sources of docs based on weights
        self.fitted_list = self.fit(final_docs,
                                    weights,
                                    header=header,
                                    ignore_keys=ignore_keys)

        # Convert to appropriate format
        fitted_context_string = stringify_dict_list(header=header, dicts=self.fitted_list, ignore_keys=ignore_keys)

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
