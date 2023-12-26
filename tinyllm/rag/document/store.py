from typing import List, Optional

from tinyllm.rag.document.document import Document


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


class DocumentStore:
    def __init__(self):
        self.store = {}

    def add_docs(self,
                 docs: List[Document],
                 name: str):
        if name in self.store:
            self.store[name] += docs
        else:
            self.store[name] = docs

    def fit_store(self,
                  context_size,
                  weights: Optional[List[float]] = None) -> List[Document]:

        # If weights are not provided, distribute docs evenly
        docs_lists = list(self.store.values())

        if not weights:
            weights = [1 / len(docs_lists)] * len(docs_lists)
        else:
            assert len(weights) == len(
                docs_lists), "Length of weights must be equal to the number of document store keys."

        # Normalize weights to ensure they sum up to 1
        total_weight = sum(weights)
        normalized_weights = [weight / total_weight for weight in weights]

        # Calculate token size for each source based on weights
        token_sizes = [int(weight * context_size) for weight in normalized_weights]

        i = 0
        section_names = list(self.store.keys())
        for doc_list, token_size in zip(docs_lists, token_sizes):
            section_docs = []
            # For each doc source, count how many docs can fit into its token size
            current_tokens = 0
            for doc in doc_list:
                doc_tokens = doc.size
                if current_tokens + doc_tokens <= token_size:
                    section_docs.append(doc)
                    current_tokens += doc_tokens
                else:
                    break

            self.store[section_names[i]] = section_docs
            i += 1

    def to_string(self,
                  start_string: str = '-----SUPPORTING DOCS-----',
                  end_string: str = '-----END OF SUPPORTING DOCS-----',
                  context_size: int = None,
                  weights: [List[float]] = None,
                  ):
        # Fit the multiple sources of docs based on weights
        self.fit_store(context_size,
                       weights)

        # Convert to appropriate format
        fitted_context_string = ''
        for section_name, docs in self.store.items():
            fitted_context_string += section_name + '\n'
            fitted_context_string += f'/n '.join([doc.to_string() for doc in docs])

        # Format the final context
        formatted_context = start_string + "\n" + fitted_context_string + "\n" + end_string
        return formatted_context
