from tinyllm.helpers import get_recursive_content

FILE_LIST = [
    # "/Users/othmanezoheir/PycharmProjects/openagents/tiny-llm/tinyllm/",
    "/Users/othmanezoheir/PycharmProjects/openagents/tiny-llm/tinyllm/types.py",
    "/Users/othmanezoheir/PycharmProjects/openagents/tiny-llm/tinyllm/function.py",
    "/Users/othmanezoheir/PycharmProjects/openagents/tiny-llm/tinyllm/parallel.py",
    "/Users/othmanezoheir/PycharmProjects/openagents/tiny-llm/tinyllm/chain.py",
    "/Users/othmanezoheir/PycharmProjects/openagents/tiny-llm/tinyllm/logger.py",
    "/Users/othmanezoheir/PycharmProjects/openagents/tiny-llm/tinyllm/config.py",
    "/Users/othmanezoheir/PycharmProjects/openagents/tiny-llm/app.py"
]
CONTEXT = get_recursive_content(FILE_LIST,
                                'py')

import pyperclip

pyperclip.copy(CONTEXT)
