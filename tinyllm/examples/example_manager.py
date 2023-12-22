from tinyllm.examples.example_selector import ExampleSelector

class ExampleManager:

    def __init__(self,
                 example_selector=ExampleSelector(name='Standard example selector'),
                 constant_examples=[]):
        self.constant_examples = constant_examples
        self.example_selector = example_selector
