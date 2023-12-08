from tinyllm import standard_embedding_function
from tinyllm.functions.util.example_selector import ExampleSelector

class ExampleManager:

    def __init__(self,
                 constant_examples=[],
                 variable_examples=[],
                 **kwargs):
        self.constant_examples = constant_examples
        self.selector = ExampleSelector(
            name="Example selector",
            examples=variable_examples,
            embedding_function=kwargs.get('embedding_function', standard_embedding_function),
            is_traced=False
        )

    def __len__(self):
        return self.num_examples

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
