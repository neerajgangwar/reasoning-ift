import json
import os
import logging
import hydra
import wandb
import lightning as L
import backoff
from pathlib import Path
from torch.utils.data import DataLoader
from lightning.pytorch.strategies import FSDPStrategy, DDPStrategy
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from omegaconf import DictConfig, OmegaConf, ListConfig
from src.data import (
    GSM8kDataset,
    GSM8kGeneratedDataset,
    ZeroShotGSM8kDataset,
)
from src.model import T5ForMath


logger = logging.getLogger(__name__)


@backoff.on_exception(backoff.expo, Exception, max_tries=10)
def load_wandb_logger(args):
    # wandb
    wandbfile = os.path.join(args.save_path, 'wandb.yaml')
    rank = int(os.environ.get('RANK', '0'))
    if rank != 0 or os.path.exists(wandbfile):
        with open(wandbfile, 'r') as f:
            wandbconfig = json.load(f)
    else:
        wandbconfig = {
            'id': wandb.util.generate_id(),
            'name': args.run_name,
        }
        with open(wandbfile, 'w') as f:
            json.dump(wandbconfig, f, indent=4)

    return L.pytorch.loggers.WandbLogger(
        id = wandbconfig['id'],
        name = wandbconfig['name'],
        save_dir = args.save_path,
        project = args.project_name,
        log_model = False,
    )


@hydra.main(version_base=None, config_path='src/conf/train', config_name='config')
def main(args : DictConfig):
    logger.info(f'Config:\n{OmegaConf.to_yaml(args)}')

    Path(args.save_path).mkdir(parents=True, exist_ok=True)

    # Check save dir
    argsfile = os.path.join(args.save_path, 'args.yaml')
    if os.path.exists(argsfile):
        logger.info('Loading arguments from the saved file.')
        args = OmegaConf.load(argsfile)
    else:
        OmegaConf.save(args, argsfile)

    wandb_logger = load_wandb_logger(args)

    # Set seed
    if args.seed is not None:
        L.seed_everything(args.seed, workers=True)

    # Checkpointing
    checkpoint_dir = os.path.join(args.save_path, 'saved_models')
    Path(checkpoint_dir).mkdir(exist_ok=True)

    best_checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename='best',
        monitor='val/accuracy',
        mode='max',
        save_last=False,
        verbose=True,
        save_weights_only=True,
    )
    last_checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        every_n_epochs=1,
        filename='last',
        verbose=True,
    )
    lr_monitor_callback = LearningRateMonitor(logging_interval='step')

    # Check if checkpoints exist
    ckpt_path = None
    if os.path.exists(os.path.join(checkpoint_dir, 'last.ckpt')):
        ckpt_path = os.path.join(checkpoint_dir, 'last.ckpt')

    # Model
    if args.model_type == 't5':
        model = T5ForMath(
            model_name_or_path=args.model_name_or_path,
            tokenizer_name_or_path=args.model_name_or_path,
            optim=args.optim,
            optim_args=args.optim_args,
            max_input_len=args.max_input_len,
            max_new_tokens=args.max_new_tokens,
            temperature=None,
            top_p=None,
            num_return_sequences=None,
            warmup_steps=args.warmup_steps,
        )
    else:
        raise NotImplementedError(f'Script is not implemented for model_type "{args.model_type}"')

    # Trainer
    if args.strategy == 'fsdp':
        strategy = FSDPStrategy(auto_wrap_policy=model.fsdp_auto_wrap_policy())
    elif args.strategy == 'ddp':
        strategy = DDPStrategy(find_unused_parameters=False)
    else:
        raise NotImplementedError(f'strategy "{args.strategy}" is not implemented')

    if isinstance(strategy, FSDPStrategy):
        assert args.gradient_clip_val is None, f'Gradient clipping is not supported with FSDP'

    trainer = L.Trainer(
        accumulate_grad_batches=args.grad_accum_steps,
        gradient_clip_val=args.gradient_clip_val,
        accelerator='auto',
        strategy=strategy,
        precision=args.precision,
        max_epochs=args.num_epochs,
        logger=[wandb_logger],
        callbacks=[best_checkpoint_callback, last_checkpoint_callback, lr_monitor_callback],
        deterministic=True,
        val_check_interval=args.val_check_interval,
        log_every_n_steps=args.log_every_n_steps,
    )

    # data
    assert args.train_batch_size % (trainer.num_devices * args.grad_accum_steps) == 0
    train_batch_size = int(args.train_batch_size / (trainer.num_devices * args.grad_accum_steps))
    val_batch_size = round(args.val_batch_size / trainer.num_devices)
    logger.info(f'Per device batch sizes | train: {train_batch_size}, val: {val_batch_size}')

    assert not isinstance(args.dataset, ListConfig)

    train_dataset = ZeroShotGSM8kDataset(
        dataset=GSM8kGeneratedDataset(
            filepath=args.dataset.train_file,
            num_examples=args.num_train_examples,
        ),
        data_mode=args.dataset.data_mode,
    )
    val_dataset = ZeroShotGSM8kDataset(
        dataset=GSM8kDataset(
            filepath=args.dataset.val_file,
            num_examples=args.num_val_examples,
        ),
        data_mode=args.dataset.data_mode,
    )
    val_file = args.dataset.val_file

    train_dataloader = DataLoader(
        train_dataset,
        collate_fn=model.collate_fn,
        batch_size=train_batch_size,
        shuffle=True,
    )
    val_dataloader = DataLoader(
        val_dataset,
        collate_fn=model.collate_fn,
        batch_size=val_batch_size,
        shuffle=False,
    )

    # Train
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader, ckpt_path=ckpt_path)

    # Save best validation accuracy
    with open(os.path.join(args.save_path, 'validation.json'), 'w') as f:

        json.dump({
            'filepath': val_file,
            'n_examples': len(val_dataset),
            'val/accuracy': trainer.checkpoint_callback.best_model_score.item(),
        }, f, indent=4)


if __name__ == '__main__':
    main()
