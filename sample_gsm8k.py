import os
import re
import pickle
import hydra
import logging
from lightning import seed_everything
from pathlib import Path
from tqdm import tqdm
from omegaconf import DictConfig
from src.data import GSM8kDataset, StaticFewShotGSM8kDataset
from src.generation import GenerationModel, MistralModel
from src.utils import chunks
from src.utils.gsm_solver import extract_cot_answer


logger = logging.getLogger(__name__)


def process(conf: DictConfig, model: GenerationModel):
    local_rank = int(os.environ.get('LOCAL_RANK', 0))

    if local_rank == 0:
        Path(conf.save_path).parent.mkdir(parents=True, exist_ok=True)
        Path(conf.save_path).touch(exist_ok=conf.overwrite_save_path)

    if conf.prompt_type == 'cot':
        extract_answer = extract_cot_answer
    else:
        raise NotImplementedError(f'prompt_type "{conf.prompt_type}" is not supported')

    data_filepath = conf.dataset[conf.split]
    logger.info(f'Loading the dataset from {data_filepath}')
    dataset = StaticFewShotGSM8kDataset(
        dataset=GSM8kDataset(filepath=data_filepath, num_examples=conf.num_examples),
        data_mode=conf.prompt_type,
    )
    logger.info(f'Number of examples to process: {len(dataset)}')

    prompts = [e['input'] for e in dataset]
    questions = [e['question'] for e in dataset]
    gold_answers = [e['output'] for e in dataset]
    targets = [e['target'] for e in dataset]

    prompt_batches = list(chunks(prompts, conf.batch_size))
    gen_answers_list = []
    for prompt_batch in tqdm(prompt_batches, desc='Generation'):
        output = model.generate(
            do_sample=True,
            prompts=prompt_batch,
            temperature=conf.temperature,
            top_p=conf.top_p,
            max_gen_len=conf.max_new_tokens,
            num_return_sequences=conf.num_return_sequences,
        )
        output = list(chunks(output, conf.num_return_sequences))
        gen_answers_list.extend(output)

    assert len(questions) == len(gold_answers) == len(gen_answers_list)
    assert all(len(gen_answers) == conf.num_return_sequences for gen_answers in gen_answers_list)

    if local_rank == 0:
        # Save before postprocessing
        save_path_pre = f'{conf.save_path}.pre'
        with open(save_path_pre, 'wb') as f:
            pickle.dump({
                'questions': questions,
                'prompts': prompts,
                'gold_answers': gold_answers,
                'gen_answers_list': gen_answers_list,
                'targets': targets,
            }, f)

        # Postprocessing
        generated_examples = []
        for question, prompt, gold_answer, target, gen_answers in tqdm(zip(questions, prompts, gold_answers, targets, gen_answers_list), desc='Postprocessing'):
            for gen_answer in gen_answers:
                answer_generated, gen_answer_clean = extract_answer(gen_answer)
                if answer_generated:
                    is_correct = gen_answer_clean == target
                else:
                    is_correct = False
                example = {
                    'question': question,
                    'prompt': prompt,
                    'gold_answer': gold_answer,
                    'gen_answer': gen_answer,
                    'gen_answer_clean': gen_answer_clean,
                    'is_correct': is_correct,
                }
                generated_examples.append(example)

        with open(conf.save_path, 'wb') as f:
            pickle.dump(generated_examples, f)

        # Remove temp file saved before postprocessing
        os.remove(save_path_pre)


@hydra.main(version_base=None, config_path='src/conf/sample', config_name='gsm8k')
def main(conf: DictConfig):
    logging.info(f'conf: {conf}')
    if conf.seed is not None:
        seed_everything(conf.seed)

    if conf.model.model_type == 'mistral':
        logger.info(f'Loding {conf.model.model_name}')
        model = MistralModel(model_name=conf.model.model_name)
    else:
        raise NotImplementedError(f'Not implemented for model type "{conf.model_type}"')

    process(conf=conf, model=model)


if __name__ == '__main__':
    main()
