import torch 
import torch.nn as nn 
import sys
sys.path.append("/Users/xingzhaohu/Downloads/code/python/ml/ml_code/bert/bert_seq2seq")
from torch.optim import Adam
import pandas as pd
import numpy as np
import os
import json
import time
import bert_seq2seq
from bert_seq2seq.tokenizer import Tokenizer, load_chinese_base_vocab
from bert_seq2seq.utils import load_bert, load_model_params, load_recent_model

auto_title_model = "./state_dict/bert_model_poem.bin"

if __name__ == "__main__":
    vocab_path = "./state_dict/roberta_wwm_vocab.txt"  # roberta模型字典的位置
    model_name = "roberta"  # 选择模型名字
    # model_path = "./state_dict/bert-base-chinese-pytorch_model.bin"  # roberta模型位
    # 加载字典
    word2idx, keep_tokens = load_chinese_base_vocab(vocab_path, simplfied=True)
    # 定义模型
    bert_model = load_bert(word2idx, model_name=model_name)
    load_model_params(bert_model, "./state_dict/roberta_wwm_pytorch_model.bin", keep_tokens=keep_tokens)

    for name, params in bert_model.named_parameters():
        print(name)


