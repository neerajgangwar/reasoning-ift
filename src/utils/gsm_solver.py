import re
import logging
from typing import Tuple, Union
from src.data.constants import NUMBER_REGEX, COT_ANSWER_PREFIX


logger = logging.getLogger(__name__)


def extract_cot_answer(answer: str) -> Tuple[bool, Union[float, None]]:
    answer = answer.lower()
    preds = answer.split(COT_ANSWER_PREFIX.lower())
    answer_flag = True if len(preds) > 1 else False
    if answer_flag:
        # Pick first answer with flag
        pred = preds[1]
    else:
        # Pick last number without flag
        pred = preds[-1]

    pred = pred.replace(',', '')
    pred = [s for s in re.findall(NUMBER_REGEX, pred)]

    if len(pred) == 0:
        return False, None

    if answer_flag:
        # choose the first element in list
        pred = pred[0]
    else:
        # choose the last element in list
        pred = pred[-1]

    # (For arithmetic tasks) if a word ends with period, it will be omitted ...
    if pred[-1] == '.':
        pred = pred[:-1]

    return True, float(pred)
