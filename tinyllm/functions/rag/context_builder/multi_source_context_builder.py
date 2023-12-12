from typing import List, Dict, Optional

from tinyllm.functions.rag.context_builder.context_builder import ContextBuilder
from tinyllm.functions.rag.document import Document


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
    def __init__(self,
                 **kwargs):
        super().__init__(**kwargs)
        self.fitted_doc_list = None

    def get_context(self,
                    docs: List[Document],
                    weights: Optional[List[float]] = None,
                    ):
        # Remove duplicates
        final_docs = remove_duplicate_dicts(docs)

        # Fit the multiple sources of docs based on weights
        self.fitted_doc_list = self.fit(final_docs,
                                        weights)

        # Convert to appropriate format
        fitted_context_string = '/n'.join([doc.format() for doc in self.fitted_doc_list])

        # Format the final context
        formatted_context = self.format_context(fitted_context_string)

        return formatted_context

    def fit(self,
            docs: List,
            weights: Optional[List[float]] = None,
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
                doc_tokens = doc.size
                if current_tokens + doc_tokens <= token_size:
                    fitted_docs.append(doc)
                    current_tokens += doc_tokens
                else:
                    break
        return fitted_docs
