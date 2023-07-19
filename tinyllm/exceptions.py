
# Prompt exceptions
class InvalidPromptInputException(Exception):
    pass

class PromptSectionValidationException(Exception):
    pass

class UserInputValidationException(Exception):
    pass


class UserInputMissingException(Exception):
    pass


# Chain exceptions
class ChainOutputValidationException(Exception):
    pass


class InvalidChainOutputException(ValueError):
    pass

class OutputParsingException(ValueError):
    pass
