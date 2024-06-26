from enum import IntEnum


class States(IntEnum):
    INIT = 1
    INPUT_VALIDATION = 2
    RUNNING = 3
    OUTPUT_VALIDATION = 4
    OUTPUT_EVALUATION = 5
    PROCESSING_OUTPUT = 6
    PROCESSED_OUTPUT_VALIDATION = 7
    PROCESSED_OUTPUT_EVALUATION = 8
    CLOSING = 9
    COMPLETE = 10
    FAILED = 11


TERMINAL_STATES = [States.COMPLETE, States.FAILED]

ALLOWED_TRANSITIONS = {
    None: [States.INIT],

    States.INIT: [States.INPUT_VALIDATION, States.FAILED],
    States.INPUT_VALIDATION: [States.RUNNING, States.FAILED],

    States.RUNNING: [States.OUTPUT_VALIDATION, States.FAILED],

    States.OUTPUT_VALIDATION: [States.COMPLETE, States.OUTPUT_VALIDATION, States.PROCESSING_OUTPUT,
                               States.OUTPUT_EVALUATION, States.FAILED],
    # Can transition to itself in the case of a streaming function
    States.OUTPUT_EVALUATION: [States.COMPLETE, States.PROCESSING_OUTPUT, States.PROCESSING_OUTPUT, States.FAILED],

    States.PROCESSING_OUTPUT: [
        States.PROCESSED_OUTPUT_VALIDATION,
        States.CLOSING,
        States.FAILED,
        States.COMPLETE
    ],
    States.PROCESSED_OUTPUT_VALIDATION: [
        States.PROCESSED_OUTPUT_EVALUATION,
        States.COMPLETE,
        States.FAILED,
    ],
    States.PROCESSED_OUTPUT_EVALUATION: [States.CLOSING, States.COMPLETE, States.FAILED],
    States.CLOSING: [States.COMPLETE, States.FAILED],

    States.COMPLETE: [States.INPUT_VALIDATION],
    States.FAILED: [States.INPUT_VALIDATION]
}
