from typing import Sequence
import torch
from accelerate import Accelerator
from transformers import AutoModelForCausalLM, AutoTokenizer
from .base import GenerationModel


class MistralModel(GenerationModel):
    def __init__(self, model_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map='auto')
        self.accelerator = Accelerator()


    def generate(
        self,
        prompts: Sequence[str],
        max_gen_len: int,
        do_sample: bool,
        temperature: float,
        top_p: float,
        num_return_sequences: int,
    ):
        tokenized = self.tokenizer(prompts, padding=True, return_tensors='pt')
        tokenized = {k : v.to(self.accelerator.device) for k, v in tokenized.items()}
        output = self.model.generate(
            **tokenized,
            max_new_tokens=max_gen_len,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            num_return_sequences=num_return_sequences
        )
        prompt_seq_len = tokenized['input_ids'].size(1)
        return self.tokenizer.batch_decode(output[:, prompt_seq_len:], skip_special_tokens=True)
