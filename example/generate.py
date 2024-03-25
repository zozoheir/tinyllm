from tinyllm.llms.tiny_function import tiny_function
from tinyllm.util.prompt_util import get_files_content
from tinyllm.validator import Validator


class ExampleModel(Validator):
    example_python: str

@tiny_function(output_model=ExampleModel, model_kwargs={'model': 'gpt-4-turbo-preview'})
async def generate_example(files_content: str):
    """
    <system>
    ROLE
    You are a Python documentation expert.
    TASK
    You will write an example for the tinyllm library, a Python library for Prompt engineering and developing Large Language Models applications in Production.
    SPECS
    The example should be for a finance workflow that extracts a Risk Score between 0 and 1 for a Credit Card application.
    The application has 2 inputs:
    - bank account history
    - employment history
    Generate random data for these arguments and pass them to the function

    The example must be a single .py file that contains everything needed to run the workflow.

    </system>

    <prompt>
    Use the files below to generate the example workflow for the tinyllm library.
    {files_content}
    </prompt>

    """


if __name__ == "__main__":
    import asyncio
    import os

    files = [os.getcwd()+'/tinyllm/tests/test_tiny_function.py',
             os.getcwd()+'/tinyllm/llms/tiny_function.py']
    files_content = get_files_content(files, formats=['.py'])
    example = asyncio.run(generate_example(files_content=files_content))
    with open('credit_risk_example.py', 'w') as f: f.write(example['output'].example_python)