import torch 
import sys
sys.path.append("/Users/xingzhaohu/Downloads/code/python/ml/ml_code/bert/bert_seq2seq")
from bert_seq2seq.tokenizer import Tokenizer, load_chinese_base_vocab
import os
import json
import time
import bert_seq2seq
from bert_seq2seq.utils import load_bert

vocab_path = "./state_dict/roberta_wwm_vocab.txt" # roberta模型字典的位置

model_name = "roberta" # 选择模型名字
model_path = "./state_dict/bert_math_ques_model.bin"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    vocab_path = "./state_dict/roberta_wwm_vocab.txt"  # roberta模型字典的位置
    model_name = "roberta"  # 选择模型名字
    # 加载字典
    word2idx = load_chinese_base_vocab(vocab_path, simplfied=False)
    tokenizer = Tokenizer(word2idx)
    # 定义模型
    bert_model = load_bert(word2idx, model_name=model_name, model_class="seq2seq")
    bert_model.set_device(device)
    bert_model.eval()
    ## 加载训练的模型参数～
    bert_model.load_all_params(model_path=model_path, device=device)
    test_data = ["王艳家买了一台洗衣机和一台电冰箱，一共花了6000元，电冰箱的价钱是洗衣机的3/5，求洗衣机的价钱．",
                 "六1班原来男生占总数的2/5，又转来5名男生，现在男生占总数的5/11，女生有多少人？", 
                 "两个相同的数相乘，积是3600，这个数是多少.",
                 "1加1等于几"]
    for text in test_data:
        with torch.no_grad():
            print(bert_model.generate(text, beam_size=3))