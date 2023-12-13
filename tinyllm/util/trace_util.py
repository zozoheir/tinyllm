import datetime as dt

from langfuse.model import UpdateGeneration, Usage, CreateGeneration, CreateSpan, UpdateSpan

from tinyllm.functions.util.helpers import count_tokens


def langfuse_generation(func):
    async def wrapper(*args, **kwargs):
        self = args[0]

        if kwargs.get('parent_observation', None):
            observation = kwargs['parent_observation']
        else:
            observation = self.trace

        generation = observation.generation(CreateGeneration(
            name=self.name,
            prompt=kwargs['messages'],
            startTime=dt.datetime.utcnow(),
        ))

        result = await func(*args, **kwargs)
        kwargs['self'] = self

        for evaluator in self.evaluators:
            kwargs['self'] = self
            await evaluator(output=result, function=self, observation=generation)

        response_message = result['choices'][0]['message']
        generation.update(UpdateGeneration(
            endTime=dt.datetime.utcnow(),
            completion=response_message,
            usage=Usage(promptTokens=count_tokens(kwargs['messages']), completionTokens=count_tokens(response_message)),
        ))
        return result

    return wrapper


def langfuse_generation_stream(func):
    async def wrapper(*args, **kwargs):
        self = args[0]

        if kwargs.get('parent_observation', None):
            observation = kwargs['parent_observation']
        else:
            observation = self.trace

        # Set up initial generation tracking
        generation = observation.generation(CreateGeneration(
            name=self.name,
            startTime=dt.datetime.utcnow(),
            prompt=kwargs['messages'],
        ))

        # Call the original async generator function
        async_gen = func(*args, **kwargs)

        # Track completion and function call information
        completion = ""
        function_call = {
            "name": None,
            "arguments": ""
        }

        async for value in async_gen:
            # Process each value yielded by the generator
            if value['type'] == "completion":
                completion += value['completion']
            elif value['type'] == "tool":
                if function_call['name'] is None:
                    function_call['name'] = value['completion']['name']
                function_call['arguments'] += value['completion']['arguments']

            yield value

        for evaluator in self.evaluators:
            kwargs['self'] = self
            await evaluator(output=value, function=self, observation=generation)

        # Update generation info after generator is done
        generation.update(UpdateGeneration(
            completion=value,
            endTime=dt.datetime.utcnow(),
            usage=Usage(promptTokens=count_tokens(kwargs['messages']), completionTokens=count_tokens(completion)),
        ))

    return wrapper


def langfuse_span(name=None, input_key=None, output_key=None, evaluators=[]):
    def decorator(func):
        async def wrapper(*args, **kwargs):
            self = args[0]
            span_name = self.name if name is None else name
            if kwargs.get('parent_observation', None):
                observation = kwargs['parent_observation']
            else:
                observation = self.trace

            # Extract input using input_key
            input_value = kwargs.get(input_key, None)

            # Create a span
            span = observation.span(
                CreateSpan(
                    name=span_name,
                    startTime=dt.datetime.utcnow(),
                    input=input_value
                )
            )
            self.parent_observation = span

            # Call the original function
            result = await func(*args, **kwargs)

            # Extract output using output_key
            output_value = result if output_key is None else result.get(output_key, None)

            # Update the span with endTime and output
            span.update(
                UpdateSpan(
                    endTime=dt.datetime.utcnow(),
                    output=output_value
                )
            )

            for evaluator in self.evaluators:
                kwargs['self'] = self
                await evaluator(output=output_value, function=self, observation=span)

            return result

        return wrapper

    return decorator
