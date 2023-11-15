from smartpy.utility.log_util import getLogger

logger = getLogger(__name__)

class FallbackStrategy:
    async def apply(self, func, *args, **kwargs):
        raise NotImplementedError

class KwargsChangeStrategy(FallbackStrategy):
    def __init__(self, fallback_kwargs):
        super().__init__()
        self.fallback_kwargs = fallback_kwargs

    async def apply(self, func, *args, **kwargs):
        # Only update the kwargs that are specified in fallback_kwargs
        updated_kwargs = {**kwargs, **self.fallback_kwargs}
        return await func(*args, **updated_kwargs)

class RetryStrategy(FallbackStrategy):
    def __init__(self, max_retries=3):
        self.max_retries = max_retries
    async def apply(self, func, *args, **kwargs):
        for _ in range(self.max_retries):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                logger.info(f"Retrying {func.__name__} due to {e}")
        raise Exception("Max retries reached")

def fallback_decorator(func):
    async def wrapper(self, *args, **kwargs):
        try:
            return await func(self, *args, **kwargs)
        except Exception as e:
            exception_type = type(e)
            if exception_type in self.fallback_strategies:
                strategy = self.fallback_strategies[exception_type]
                return await strategy.apply(func, self, *args, **kwargs)
            raise
    return wrapper

# Example usage
fallback_strategies_example = {
    FallbackStrategy: KwargsChangeStrategy(fallback_kwargs={'model':'gpt-4-1106-preview'}),
}