from enum import Enum


class States(Enum):
    INIT = 'init'
    READY = 'ready'
    INPUT_VALIDATION = 'input validation'
    RUNNING = 'running'
    OUTPUT_VALIDATION = 'output validation'
    PROCESSING_OUTPUT = 'processing output'
    COMPLETE = 'complete'
    FAILED = 'failed'


ALLOWED_TRANSITIONS = {
    None: [States.INIT],
    States.INIT: [States.INPUT_VALIDATION, States.FAILED],
    #States.READY: [States.INPUT_VALIDATION, States.FAILED],
    States.INPUT_VALIDATION: [States.RUNNING, States.FAILED],
    States.RUNNING: [States.OUTPUT_VALIDATION, States.FAILED],
    States.OUTPUT_VALIDATION: [States.COMPLETE, States.PROCESSING_OUTPUT, States.FAILED],
    States.PROCESSING_OUTPUT: [States.COMPLETE, States.FAILED],
    States.COMPLETE: [States.INPUT_VALIDATION],
    States.FAILED: [States.INPUT_VALIDATION]
}
