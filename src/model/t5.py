from typing import Dict, Any
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from transformers.models.t5.modeling_t5 import T5Block
from .base import HFTransformerForMath


class T5ForMath(HFTransformerForMath):
    def load_model(self, model_name_or_path: str) -> PreTrainedModel:
        return AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)


    def load_tokenizer(self, tokenizer_name_or_path: str) -> PreTrainedTokenizer:
        return AutoTokenizer.from_pretrained(tokenizer_name_or_path)


    def model_type(self) -> str:
        return 't5'


    def fsdp_auto_wrap_policy(self) -> Any:
        return {T5Block}


    def generate(self, batch: Dict[str, Any], config: Dict[str, Any]) -> float:
        output = self.model.generate(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            synced_gpus=True,
            **config,
        )
        return self.tokenizer.batch_decode(output, skip_special_tokens=True)


    def collate_fn(self, batch: Any) -> Dict[str, Any]:
        questions = [e['question'] for e in batch]
        inputs = [e['input'] for e in batch]
        answers = [e['output'] for e in batch]
        targets = [e['target'] for e in batch]
        data_modes = [e['data_mode'] for e in batch]

        input_tokenized = self.tokenizer(inputs, max_length=self.max_input_len, truncation=True, padding=True, return_tensors='pt')
        labels_tokenized = self.tokenizer(answers, max_length=self.max_new_tokens, truncation=True, padding=True, return_tensors='pt')

        return {
            'input_ids': input_tokenized['input_ids'],
            'attention_mask': input_tokenized['attention_mask'],
            'labels': labels_tokenized['input_ids'],
            'questions': questions,
            'answers': answers,
            'targets': targets,
            'data_modes': data_modes,
        }
