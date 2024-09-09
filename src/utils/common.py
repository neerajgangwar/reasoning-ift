from typing import List, Any


def chunks(lst: List[Any], n: int) -> List[List[Any]]:
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]
