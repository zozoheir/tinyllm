from enum import Enum


class States(Enum):
    INIT = 'init'
    READY = 'ready'
    INPUT_VALIDATION = 'input validation'
    RUNNING = 'running'
    COMPLETE = 'complete'
    FAILED = 'failed'


ALLOWED_TRANSITIONS = {
    None: [States.INIT],
    States.INIT: [States.INPUT_VALIDATION, States.FAILED],
    #States.READY: [States.INPUT_VALIDATION, States.FAILED],
    States.INPUT_VALIDATION: [States.RUNNING, States.FAILED],
    States.RUNNING: [States.COMPLETE, States.FAILED],
    States.COMPLETE: [],
    States.FAILED: []
}


