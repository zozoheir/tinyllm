import asyncio
import logging
import sys
import unittest

from tinyllm import langfuse_client

logger = logging.getLogger()
logger.level = logging.DEBUG
logging.basicConfig(level=logging.DEBUG)


class AsyncioTestCase(unittest.TestCase):
    def setUp(self):
        stream_handler = logging.StreamHandler(sys.stdout)
        logger.addHandler(stream_handler)
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

    def tearDown(self):
        self.loop.close()
        asyncio.set_event_loop(None)
        langfuse_client.flush()
