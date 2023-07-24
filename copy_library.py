from tinyllm.helpers import get_recursive_content

FILE_LIST = [
    # "/Users/othmanezoheir/PycharmProjects/openagents/tiny-llm/tinyllm/",
    "/Users/othmanezoheir/PycharmProjects/openagents/tiny-llm/tinyllm/functions",
    "/Users/othmanezoheir/PycharmProjects/openagents/tiny-llm/tinyllm/examples"
]
CONTEXT = get_recursive_content(FILE_LIST,
                                'py')

import pyperclip

pyperclip.copy(CONTEXT)
