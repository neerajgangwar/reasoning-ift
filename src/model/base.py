import torch
import lightning as L
from abc import ABC, abstractmethod
from typing import Dict, Any
from transformers import (
    Adafactor,
    PreTrainedModel,
    PreTrainedTokenizer,
    get_constant_schedule_with_warmup,
)
from src.utils import chunks
from src.utils.gsm_solver import extract_cot_answer


class HFTransformerForMath(L.LightningModule, ABC):
    def __init__(
        self,
        model_name_or_path: str,
        tokenizer_name_or_path: str,
        optim: str,
        optim_args: Dict[str, Any],
        max_input_len: int,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        num_return_sequences: int,
        warmup_steps: int,
    ):
        super(HFTransformerForMath, self).__init__()
        self.model_name_or_path = model_name_or_path
        self.tokenizer_name_or_path = tokenizer_name_or_path
        self.optim = optim
        self.optim_args = optim_args
        self.max_input_len = max_input_len
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.num_return_sequences = num_return_sequences
        self.warmup_steps = warmup_steps

        # self.model = self.load_model(self.model_name_or_path)
        self.model = None
        self.tokenizer = self.load_tokenizer(self.tokenizer_name_or_path)

        self.save_hyperparameters()


    def configure_model(self) -> None:
        super().configure_model()

        if self.model is not None:
            return

        self.model = self.load_model(self.model_name_or_path)
        # This is required by Lightning 2.2.0 (https://github.com/Lightning-AI/pytorch-lightning/releases/tag/2.2.0#highlights-eval).
        # `training_mode` is False for many layers in HF models.
        self.model.train()
        assert self.model.config.model_type == self.model_type(), f'Incorrect model_type in config: "{self.model.model_type}"'


    @abstractmethod
    def load_model(self, model_name_or_path: str) -> PreTrainedModel:
        pass


    @abstractmethod
    def load_tokenizer(self, tokenizer_name_or_path: str) -> PreTrainedTokenizer:
        pass


    @abstractmethod
    def model_type(self) -> str:
        pass


    @abstractmethod
    def fsdp_auto_wrap_policy(self) -> Any:
        pass


    @abstractmethod
    def generate(self, batch: Dict[str, Any], config: Dict[str, Any]) -> float:
        pass


    @abstractmethod
    def collate_fn(self, batch: Any, is_test: bool) -> Dict[str, Any]:
        pass


    def configure_optimizers(self):
        if self.optim == 'AdamW':
            optimizer = torch.optim.AdamW(self.model.parameters(), **self.optim_args)
        elif self.optim == 'AdaFactor':
            optimizer = Adafactor(
                self.model.parameters(),
                scale_parameter=False,
                relative_step=False,
                clip_threshold=1.0,
                decay_rate=-0.8,
                lr=self.optim_args['lr'],
                weight_decay=self.optim_args['weight_decay'],
            )
        else:
            raise NotImplementedError(f'{__class__} is not implemented for optim "{self.optim}"')

        if self.warmup_steps > 0:
            lr_schedule = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=self.warmup_steps)
            return [optimizer], [{'scheduler': lr_schedule, 'interval': 'step'}]
        else:
            return optimizer


    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> float:
        batch_size = batch['input_ids'].size(0)
        output = self.model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels'],
            return_dict=True,
        )
        loss = output['loss']
        self.log('train/loss', loss, batch_size=batch_size, on_step=True, on_epoch=False, sync_dist=True)
        return loss


    def eval_model(self, batch: Dict[str, Any], config: Dict[str, Any]) -> float:
        decoded_output = self.generate(batch, config)
        if 'num_return_sequences' in config:
            decoded_output = list(chunks(decoded_output, config['num_return_sequences']))
        else:
            decoded_output = list(chunks(decoded_output, 1))

        assert len(decoded_output) == len(batch['targets']), f'length mismatch ({len(decoded_output)}, {len(batch["targets"])})'

        total, correct, model_outputs = 0, 0, []
        for target, preds, data_mode in zip(batch['targets'], decoded_output, batch['data_modes']):
            total += 1
            pred_answers = []
            if data_mode == 'cot':
                extract_answer = extract_cot_answer
            else:
                raise NotImplementedError(f'data_mode "{data_mode}" is not supported')

            for pred in preds:
                success, pred_answer = extract_answer(pred)
                if success:
                    pred_answers.append(pred_answer)

            if len(pred_answers) > 0:
                maj_answer = max(set(pred_answers), key=pred_answers.count)
                is_correct = float(maj_answer) == target
                correct += int(is_correct)
            else:
                is_correct = False

            model_outputs.append({
                'preds': preds,
                'is_correct': is_correct,
            })

        return {
            'total': total,
            'correct': correct,
            'accuracy': correct / total,
            'model_outputs': model_outputs,
        }


    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        batch_size = batch['input_ids'].size(0)
        config = {
            'do_sample': False,
            'num_beams': 1,
            'max_new_tokens': self.max_new_tokens
        }
        output = self.eval_model(batch, config)
        self.log('val/accuracy', output['accuracy'], batch_size=batch_size, on_step=False, on_epoch=True, sync_dist=True)


    def on_test_start(self) -> None:
        self.model_outputs = []


    def test_step(self, batch: Dict[str, Any], batch_idx: int, dataloader_idx: int=0) -> None:
        batch_size = batch['input_ids'].size(0)
        # Greedy decoding
        config = {
            'do_sample': False,
            'num_beams': 1,
            'max_new_tokens': self.max_new_tokens,
        }
        greedy_output = self.eval_model(batch, config)
        self.log('test/greedy_accuracy', greedy_output['accuracy'], batch_size=batch_size, on_step=False, on_epoch=True, sync_dist=True)

        # Majority voting
        config = {
            'do_sample': True,
            'temperature': self.temperature,
            'top_p': self.top_p,
            'num_return_sequences': self.num_return_sequences,
            'max_new_tokens': self.max_new_tokens,
        }
        sampling_output = self.eval_model(batch, config)
        self.log(f'test/maj1@{self.num_return_sequences}', sampling_output['accuracy'], batch_size=batch_size, on_step=False, on_epoch=True, sync_dist=True)

        self.log('test/dataset_size', batch_size, batch_size=batch_size, on_step=False, on_epoch=True, sync_dist=True, reduce_fx='sum')

        greedy_model_outputs = greedy_output['model_outputs']
        sampling_model_outputs = sampling_output['model_outputs']
        questions = batch['questions']
        targets = batch['targets']
        inputs = self.tokenizer.batch_decode(batch['input_ids'], skip_special_tokens=True)
        for question, input, target, greedy_model_output, sampling_model_output in zip(questions, inputs, targets, greedy_model_outputs, sampling_model_outputs):
            assert len(greedy_model_output['preds']) == 1, (len(greedy_model_output['preds']))
            assert len(sampling_model_output['preds']) == self.num_return_sequences, (len(sampling_model_output['preds']), self.num_return_sequences)
            self.model_outputs.append({
                'question': question,
                'input': input,
                'target': target,
                'greedy_pred': greedy_model_output['preds'][0],
                'sampling_pred': sampling_model_output['preds'],
                'greedy_is_correct': greedy_model_output['is_correct'],
                'sampling_is_correct': sampling_model_output['is_correct'],
            })
