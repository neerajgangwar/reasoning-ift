import json
import os
import logging
import hydra
import gzip
import lightning as L
from pathlib import Path
from functools import partial
from torch.utils.data import DataLoader, Dataset
from omegaconf import DictConfig, OmegaConf
from lightning.pytorch.strategies import FSDPStrategy, DDPStrategy
from src.data import (
    GSM8kDataset,
    ZeroShotGSM8kDataset,
    MultiArithDataset,
    SVAMPDataset,
    ASDivDataset,
    BigBenchDataset,
)
from src.model import T5ForMath


logger = logging.getLogger(__name__)


def create_dataset(args: DictConfig, data_mode: str) -> Dataset:
    if args.dataset.dataset_name == 'gsm8k':
        test_dataset = ZeroShotGSM8kDataset(
            dataset=GSM8kDataset(
                filepath=args.dataset.test_file,
                num_examples=args.num_test_examples,
            ),
            data_mode=data_mode,
        )
    elif args.dataset.dataset_name == 'multiarith':
        test_dataset = MultiArithDataset(args.dataset.split, data_mode=data_mode, few_shot_input=False)
    elif args.dataset.dataset_name == 'svamp':
        test_dataset = SVAMPDataset(args.dataset.filepath, data_mode=data_mode, few_shot_input=False)
    elif args.dataset.dataset_name == 'asdiv':
        test_dataset = ASDivDataset(args.dataset.filepath, data_mode=data_mode, few_shot_input=False)
    elif args.dataset.dataset_name == 'bigbench':
        test_dataset = BigBenchDataset(args.dataset.filepaths, data_mode=data_mode, few_shot_input=False)
    else:
        raise NotImplementedError(f'This script is not implemented for the "{args.dataset.dataset_name}" dataset')

    return test_dataset


@hydra.main(version_base=None, config_path='src/conf/test', config_name='config')
def main(args : DictConfig):
    logger.info(f'Config:\n{args}')

    assert os.path.exists(args.saved_model_path)

    train_args_file = os.path.join(args.saved_model_path, 'args.yaml')
    assert os.path.exists(train_args_file)
    train_args = OmegaConf.load(train_args_file)

    # Set seed
    if args.seed is not None:
        L.seed_everything(args.seed, workers=True)

    ckpt_path = os.path.join(args.saved_model_path, f'saved_models/{args.ckpt_name}.ckpt')
    assert os.path.exists(ckpt_path), f'{ckpt_path} does not exists'

    results_filepath = os.path.join(args.saved_model_path, 'tests', args.results_filename)
    assert results_filepath.endswith('.json.gz'), f'results_filename must end with .json.gz, provided {args.results_filename}'
    Path(results_filepath).parent.mkdir(parents=True, exist_ok=True)
    Path(results_filepath).touch(exist_ok=False)

    # Check if validation file exists to ensure training is complete
    val_filepath = os.path.join(args.saved_model_path, 'validation.json')
    assert args.skip_val_file_check or (os.path.exists(val_filepath) and os.stat(val_filepath).st_size != 0), f'Failed val file check: {val_filepath}'

    # Model
    if train_args.model_type == 't5':
        model = T5ForMath.load_from_checkpoint(
            ckpt_path,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            num_return_sequences=args.num_return_sequences,
        )
    else:
        raise NotImplementedError(f'Script not implemented for model_type "{train_args.model_type}"')

    # Trainer
    if train_args.strategy == 'fsdp':
        strategy = FSDPStrategy(auto_wrap_policy=model.fsdp_auto_wrap_policy())
    elif train_args.strategy == 'ddp':
        strategy = DDPStrategy(find_unused_parameters=False)
    else:
        raise NotImplementedError(f'strategy "{args.strategy}" is not implemented')
    trainer = L.Trainer(
        accelerator='auto',
        devices=1,
        num_nodes=1,
        strategy=strategy,
        precision=args.precision,
        logger=False,
        deterministic='warn',
    )

    # data
    test_batch_size = round(args.test_batch_size / trainer.num_devices)
    logger.info(f'Per device batch size: {test_batch_size}')
    test_dataset = create_dataset(args, train_args.dataset.data_mode)
    test_dataloader = DataLoader(
        test_dataset,
        collate_fn=partial(model.collate_fn),
        batch_size=test_batch_size,
        shuffle=False,
    )

    results = trainer.test(model, dataloaders=test_dataloader)
    results = results[0]

    model_outputs = model.model_outputs
    model.model_outputs = None

    assert len(model_outputs) == results['test/dataset_size'], (len(model_outputs), results['test/dataset_size'])

    with gzip.open(results_filepath, 'wt') as f:
        json.dump({
            'args': OmegaConf.to_container(args),
            'results': results,
            'model_outputs': model_outputs,
        }, f, indent=4)


if __name__ == '__main__':
    main()
