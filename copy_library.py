from tinyllm.helpers import get_recursive_content

FILE_LIST = [
    #"/Users/othmanezoheir/PycharmProjects/openagents/tiny-llm/tinyllm/",
    "/Users/othmanezoheir/PycharmProjects/openagents/tiny-llm/tinyllm/function.py",
    "/Users/othmanezoheir/PycharmProjects/openagents/tiny-llm/tinyllm/parallel.py",
    "/Users/othmanezoheir/PycharmProjects/openagents/tiny-llm/tinyllm/chain.py"
]
CONTEXT = get_recursive_content(FILE_LIST,
                                'py')

import pyperclip
pyperclip.copy(CONTEXT)