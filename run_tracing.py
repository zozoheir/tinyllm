import time
from datetime import datetime

from langfuse.model import CreateTrace, CreateGeneration, UpdateGeneration

from tinyllm import langfuse_client

trace = langfuse_client.trace(CreateTrace(name='test'))

for i in range(10):
    generation = trace.generation(CreateGeneration(
        name=f"gen {i}",
        prompt='output',
        startTime=datetime.now(),
        metadata={'api_result': i}
    ))
    time.sleep(1)
    generation.update(UpdateGeneration(
        name=f"gen {i}",
        endTime=datetime.now(),
        completion='output',
        metadata={'api_result': i}
    ))
