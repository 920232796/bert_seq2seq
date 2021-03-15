import sys
sys.path.append("/Users/xingzhaohu/Downloads/code/python/ml/ml_code/bert/bert_seq2seq")

import torch
from bert_seq2seq.utils import load_gpt
from bert_seq2seq.tokenizer import load_chinese_base_vocab
from transformers import AutoTokenizer
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_path = "./state_dict/gpt_auto_story.bin"

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("pranavpsv/gpt2-genre-story-generator")
    word2ix = tokenizer.get_vocab()
    model = load_gpt(word2ix, tokenizer=tokenizer)
    model.eval()
    model.set_device(device)
    model.load_all_params(model_path)

    print(model.sample_generate_english("Strong Winds", out_max_length=300, add_eos=True))