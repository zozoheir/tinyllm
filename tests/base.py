import asyncio
import unittest

class AsyncioTestCase(unittest.TestCase):
    def setUp(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

    def tearDown(self):
        self.loop.close()
        asyncio.set_event_loop(None)

    def run_async(self, coro):
        return self.loop.run_until_complete(coro)
