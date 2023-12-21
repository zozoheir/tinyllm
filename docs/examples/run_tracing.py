import datetime as dt
import time

from tinyllm import langfuse_client

trace = langfuse_client.trace(
    name='test trace',
    start_time=dt.datetime.now(),
)
span = trace.span(
    name='test span',
    start_time=dt.datetime.now(),
)
time.sleep(1)
span.update(
    end_time=dt.datetime.now(),
)
trace.update(
    end_time=dt.datetime.now(),
)
