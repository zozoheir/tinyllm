from tinyllm.helpers import get_recursive_content

FILE_LIST = [
    #"/Users/othmanezoheir/PycharmProjects/openagents/tiny-llm/tinyllm/",
    "/Users/othmanezoheir/PycharmProjects/openagents/tiny-llm/tinyllm/operator.py"
    # '/Users/othmanezoheir/PycharmProjects/openagents/openagents-backend/prompts/rest_api_prompting.py'
]
CONTEXT = get_recursive_content(FILE_LIST,
                                'py')

import pyperclip
pyperclip.copy(CONTEXT)