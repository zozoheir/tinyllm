import asyncio
import logging
import sys
import unittest

from tinyllm import langfuse_client, tinyllm_config


class AsyncioTestCase(unittest.TestCase):
    def setUp(self):
        # Tests are run in live mode
        self.loop = asyncio.get_event_loop()
        asyncio.set_event_loop(self.loop)

    def tearDown(self):
        self.loop.close()
        asyncio.set_event_loop(None)
        langfuse_client.flush()

