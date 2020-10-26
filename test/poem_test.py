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
    bert_model.eval()
#     ## 加载预训练的模型参数～
    # load_model_params(bert_model, model_path)
    checkpoint = torch.load(auto_title_model, map_location="cpu")
    # print(checkpoint)
    bert_model.load_state_dict(torch.load(auto_title_model, map_location="cpu"), strict=False)
    test_data = ["天涯海角##七言绝句"]
#     #  test_data = [
# #               "本文总结了十个可穿戴产品的设计原则而这些原则同样也是笔者认为是这个行业最吸引人的地方1为人们解决重复性问题2从人开始而不是从机器开始3要引起注意但不要刻意4提升用户能力而不是取代人",
# #              "2007年乔布斯向人们展示iPhone并宣称它将会改变世界还有人认为他在夸大其词然而在8年后以iPhone为代表的触屏智能手机已经席卷全球各个角落未来智能手机将会成为真正的个人电脑为人类发展做出更大的贡献", 
# #              "雅虎发布2014年第四季度财报并推出了免税方式剥离其持有的阿里巴巴集团15％股权的计划打算将这一价值约400亿美元的宝贵投资分配给股东截止发稿前雅虎股价上涨了大约7％至5145美元", 
#                 # "新华社受权于18日全文播发修改后的《中华人民共和国立法法》修改后的立法法分为“总则”“法律”“行政法规”“地方性法规自治条例和单行条例规章”“适用与备案审查”“附则”等6章共计105条"]
    for text in test_data:
        print(bert_model.generate(text, beam_size=3, is_poem=True))

        # print(name[0])


