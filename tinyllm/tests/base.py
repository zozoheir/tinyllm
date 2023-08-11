import asyncio
import unittest

from tinyllm.llm_trace import langfuse_client


class AsyncioTestCase(unittest.TestCase):
    def setUp(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

    def tearDown(self):
        self.loop.close()
        asyncio.set_event_loop(None)
        langfuse_client.flush()
