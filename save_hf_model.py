import argparse
import os
from src.model import T5ForMath


def save_hf_model(ckpt_path, hf_save_path):
    print(f'ckpt_path: {ckpt_path}\nhf_save_path: {hf_save_path}')
    model = T5ForMath.load_from_checkpoint(ckpt_path)
    assert not os.path.exists(hf_save_path)
    model.model.save_pretrained(hf_save_path)
    model.tokenizer.save_pretrained(hf_save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Script to convert a T5ForMath model to a HF model')
    parser.add_argument('--ckpt_path', type=str, required=True)
    parser.add_argument('--hf_save_path', type=str, required=True)
    args = parser.parse_args()
    save_hf_model(
        args.ckpt_path,
        args.hf_save_path,
    )
