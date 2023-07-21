class Functions:
    OPERATOR = 'operator'
    CHAIN = 'chain'

    # APIs
    STORE = 'store'
    PROVIDER = 'provider'
    PROVIDER_CACHE = 'provider_cache'

    LLM_PROVIDER = 'llm_provider'
    LLM_STORE = 'llm_store'
    LLM = 'llm'


class Chains:
    CHAIN = 'chain'
    PARALLEL = 'parallel'


class States:
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


