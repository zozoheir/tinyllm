from enum import Enum


class States(Enum):
    INIT = 'init'
    READY = 'ready'
    INPUT_VALIDATION = 'input validation'
    RUNNING = 'running'
    OUTPUT_VALIDATION = 'output validation'
    EVALUATING = 'evaluating'
    PROCESSING_OUTPUT = 'processing output'
    PROCESSED_OUTPUT_VALIDATION = 'processed output validation'
    COMPLETE = 'complete'
    FAILED = 'failed'
    EVALUATION  = 'evaluation'


ALLOWED_TRANSITIONS = {
    None: [States.INIT],
    States.INIT: [States.INPUT_VALIDATION, States.FAILED],
    States.INPUT_VALIDATION: [States.RUNNING, States.FAILED],
    States.RUNNING: [States.OUTPUT_VALIDATION, States.FAILED, States.EVALUATING],
    States.OUTPUT_VALIDATION: [States.COMPLETE, States.PROCESSING_OUTPUT, States.EVALUATING, States.OUTPUT_VALIDATION, States.FAILED],
    States.PROCESSING_OUTPUT: [States.PROCESSED_OUTPUT_VALIDATION, States.FAILED, States.EVALUATING, States.COMPLETE],
    States.PROCESSED_OUTPUT_VALIDATION: [States.COMPLETE, States.FAILED, States.EVALUATING],
    States.EVALUATING: [States.COMPLETE, States.FAILED, States.OUTPUT_VALIDATION, States.PROCESSING_OUTPUT, States.PROCESSED_OUTPUT_VALIDATION],
    States.COMPLETE: [States.INPUT_VALIDATION],
    States.FAILED: [States.INPUT_VALIDATION]
}
