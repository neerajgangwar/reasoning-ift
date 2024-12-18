from abc import ABC, abstractmethod
from typing import Sequence


class GenerationModel(ABC):
    @abstractmethod
    def generate(
        self,
        prompts: Sequence[str],
        max_gen_len: int,
        do_sample: bool,
        temperature: float,
        top_p: float,
        num_return_sequences: int,
    ):
        pass