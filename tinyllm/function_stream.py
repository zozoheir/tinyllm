from abc import abstractmethod
from typing import Any, Optional

from tinyllm.function import Function
from tinyllm import langfuse_client, tinyllm_config
from tinyllm.state import States
from tinyllm.tracing.langfuse_context import observation
from tinyllm.validator import Validator


class DefaultFunctionStreamOutputValidator(Validator):
    streaming_status: str
    type: str  # assistant_response, tool
    delta: Optional[dict]
    completion: Any


class FunctionStream(Function):

    def __init__(self,
                 **kwargs):
        super().__init__(output_validator=DefaultFunctionStreamOutputValidator,
                         **kwargs)

    @abstractmethod
    async def run(self,
                  **kwargs):
        yield None

    @observation('span', stream=True)
    async def __call__(self, **kwargs):
        try:
            self.input = kwargs

            # Validate input
            self.transition(States.INPUT_VALIDATION)
            validated_input = self.validate_input(**kwargs)

            # Run
            self.transition(States.RUNNING)
            async for message in self.run(**validated_input):
                # Output validation
                if 'status' in message.keys():
                    if message['status'] == 'success':
                        message = message['output']
                    else:
                        raise Exception(message['message'])

                self.transition(States.OUTPUT_VALIDATION)
                self.validate_output(**message)

                yield {"status": "success",
                       "output": message}

            self.output = message

            # Process output
            self.transition(States.PROCESSING_OUTPUT)
            self.processed_output = await self.process_output(**self.output)

            # Validate processed output
            if self.processed_output_validator:
                self.validate_processed_output(**self.processed_output)

            # Evaluate processed output
            for evaluator in self.processed_output_evaluators:
                await evaluator(**self.processed_output, observation=self.observation)

            # Complete
            self.transition(States.COMPLETE)
            langfuse_client.flush()

        except Exception as e:
            output_message = await self.handle_exception(e)
            # Raise or return error
            if tinyllm_config['OPS']['DEBUG']:
                raise e
            if type(e) in self.fallback_strategies:
                raise e
            else:
                yield output_message
