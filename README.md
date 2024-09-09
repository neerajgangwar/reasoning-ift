Setup
-----
### Cloning the repository
To clone the repository and fetch the submodules
```bash
git clone --recurse-submodules <URL>
```

### Setting up the Python environment
The Python environment can be setup using `environment.yml`.
```bash
mamba env create -n ift -f environment.yml
```

GSM8k Dataset
-------------
As GSM8k does not come with a validation set, sample 512 examples from the training set to create one.
```bash
python process_gsm8k.py save_dir=<DATASET_SAVE_PATH>
```
This script takes two more options:\
1. `seed`: Control the seed for random sampling (default: 42).
2. `n_val_examples`: Validation dataset size (default: 512).

Distilling GSM8k Responses
-----------------------
To create GSM8k (Dist.), `sample_gsm8k.py` can be used.
```bash
torchrun sample_gsm8k.py \
    model=mistral \
    model.model_name=mistralai/Mistral-7B-v0.1 \
    batch_size=4 \
    do_sample=true \
    max_new_tokens=512 \
    top_p=0.9 \
    temperature=0.6 \
    num_return_sequences=8 \
    prompt_type=cot \
    save_path=<SAVE_PATH> \
    seed=42
```
This scripts serializes the data and saves it (See [`pickle`](https://docs.python.org/3/library/pickle.html)).

See [`src/conf/sample/gsm8k.yaml`](src/conf/sample/gsm8k.yaml) for all args that this script accepts.

Training on the Arithmetic Dataset
----------------------------------
To train FlanT5 on the arithmetic dataset, the following command can be used
```bash
python train_arith.py \
    model_type=t5 \
    model_name_or_path=google/flan-t5-base \
    save_path=<OUTPUT_DIR> \
    num_epochs=10 \
    run_name=<WANDB_RUN_NAME> \
    optim=AdamW \
    optim_args="{lr:1e-4,weight_decay:1e-4}" \
    max_new_tokens=512 \
    dataset=goat \
    strategy=ddp \
    gradient_clip_val=1.0 \
    train_batch_size=128 \
    grad_accum_steps=2 \
    seed=42 \
    project_name=<WANDB_PROJ_NAME>
```
W&B can be disabled by setting the environment variable `WANDB_MODE` to `disabled`.

Once the model is trained, `save_hf_model.py` can be used to save the model in the HuggingFace format. This is required if you would like to initialize the model from these checkpoints while fine-tuning FlanT5 on GSM8k.

Training on GSM8k
-----------------
To train FlanT5 on GSM8k (Orig.), the following command can be used
```bash
python train_ift.py \
    model_type=t5 \
    model_name_or_path=<PATH_TO_BASE_MODEL> \
    save_path=<OUTPUT_DIR> \
    num_epochs=20 \
    run_name=<WANDB_RUN_NAME> \
    optim=AdamW \
    optim_args="{lr:1e-4,weight_decay:1e-4}" \
    max_new_tokens=512 \
    dataset=gsm8k_orig \
    strategy=ddp \
    gradient_clip_val=1.0 \
    train_batch_size=128 \
    val_batch_size=128 \
    grad_accum_steps=4 \
    seed=42 \
    project_name=<WANDB_PROJ_NAME>
```

To train on GSM8k (Dist.), use `dataset=gsm8k_dist`.

Evaluation
----------
To evaluate the trained model, the following command can be used
```bash
python test.py \
    dataset=<DATASET> \
    saved_model_path=<MODEL_DIR_TO_TEST> \
    results_filename=<RESULTS_FILE_NAME> \
    test_batch_size=32 \
    seed=42 \
    ckpt_name=<CKPT_TO_TEST>
```

`<DATASET>` can be `gsm8k`, `multiarith`, `asdiv`, `svamp`, or `bigbench`. For the best checkpoint, `<CKPT_TO_TEST>` would be `best`. See `saved_models` under `<MODEL_DIR_TO_TEST>` for other saved checkpoints while training.
