import json
from typing import Any, Dict


def parse_block(block: str, block_language='json') -> Dict[str, Any]:
    block = block.strip().split(f'```{block_language}')[1].split('```')[0]
    if block_language == 'json':
        return json.loads(block)
    else:
        return block
