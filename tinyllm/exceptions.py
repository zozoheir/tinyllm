class CustomException(Exception):
    def __init__(self, operator, message: str):
        self.operator = operator
        super(CustomException, self).__init__(f"{operator.tag} - {message}")

class OperatorError(CustomException):
    pass

class InvalidStateTransition(CustomException):
    pass

class InvalidInput(CustomException):
    pass

class InvalidOutput(CustomException):
    pass


class MissingLLM(CustomException):
    pass
