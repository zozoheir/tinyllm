class ExampleManager:

    def __init__(self,
                 example_selector=None,
                 constant_examples=[]):
        self.constant_examples = constant_examples
        self.example_selector = example_selector
