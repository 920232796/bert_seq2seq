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
from bert_seq2seq.utils import load_bert

relation_extrac_model = "./state_dict/nezha_relation_extract.bin"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vocab_path = "./state_dict/roberta_wwm_vocab.txt"  # roberta模型字典的位置
model_name = "nezha"  # 选择模型名字
# model_path = "./state_dict/bert-base-chinese-pytorch_model.bin"  # roberta模型位
# 加载字典
word2idx = load_chinese_base_vocab(vocab_path, simplfied=False)
tokenizer = Tokenizer(word2idx)
idx2word = {v: k for k, v in word2idx.items()}

predicate2id, id2predicate = {}, {}
with open('./corpus/三元组抽取/all_50_schemas') as f:
    for l in f:
        l = json.loads(l)
        if l['predicate'] not in predicate2id:
            id2predicate[len(predicate2id)] = l['predicate']
            predicate2id[l['predicate']] = len(predicate2id)


def search(pattern, sequence):
    """从sequence中寻找子串pattern
    如果找到，返回第一个下标；否则返回-1。
    """
    n = len(pattern)
    for i in range(len(sequence)):
        if sequence[i:i + n] == pattern:
            return i
    return -1

def search_subject(token_ids, subject_labels):
    # subject_labels: (lens, 2)
    if type(subject_labels) is torch.Tensor:
        subject_labels = subject_labels.numpy()
    if type(token_ids) is torch.Tensor:
        token_ids = token_ids.cpu().numpy()
    subjects = []
    subject_ids = []
    start = -1
    end = -1
    for i in range(len(token_ids)):
        if subject_labels[i, 0] > 0.5:
            start = i
            for j in range(len(token_ids)):
                if subject_labels[j, 1] > 0.5:
                    subject_labels[j, 1] = 0
                    end = j
                    break
            if start == -1 or end == -1:
                continue
            subject = ""
            for k in range(start, end + 1):
                subject += idx2word[token_ids[k]]
            # print(subject)
            subject_ids.append([start, end])
            start = -1
            end = -1
            subjects.append(subject)

    return subjects, subject_ids

def search_object(token_ids, object_labels):
    objects = []
    if type(object_labels) is torch.Tensor:
        object_labels = object_labels.numpy()
    if type(token_ids) is torch.Tensor:
        token_ids = token_ids.cpu().numpy()
    # print(object_labels.sum())
    start = np.where(object_labels[:, :, 0] > 0.5)
    end = np.where(object_labels[:, :, 1] > 0.5)
    # print(start)
    # print(end)
    for _start, predicate1 in zip(*start):
        for _end, predicate2 in zip(*end):
            if _start <= _end and predicate1 == predicate2:
                object_text = ""
                for k in range(_start, _end + 1):
                    # print(token_ids(k))
                    object_text += idx2word[token_ids[k]]
                objects.append(
                   (id2predicate[predicate1], object_text)
                )
                break 
    
    return objects

if __name__ == "__main__":
    
    # 定义模型
    bert_model = load_bert(word2idx, model_class="relation_extrac", model_name=model_name, target_size=len(predicate2id))
    bert_model.eval()
    bert_model.set_device(device)
#   ## 加载预训练的模型参数～
    checkpoint = torch.load(relation_extrac_model, map_location="cpu")
    # print(checkpoint)
    bert_model.load_all_params(model_path=relation_extrac_model, device=device)
    text = ["查尔斯·阿兰基斯（Charles Aránguiz），1989年4月17日出生于智利圣地亚哥，智利职业足球运动员，司职中场，效力于德国足球甲级联赛勒沃库森足球俱乐部", 
            "《星空黑夜传奇》是连载于起点中文网的网络小说，作者是啤酒的罪孽",
            "《李烈钧自述》是2011年11月1日人民日报出版社出版的图书，作者是李烈钧",
            "杨铁心和郭啸天兄弟二人在牛家村的农屋里喝酒，他们的岳飞大将军在风波亭被害之事，二人希望能够像岳飞大将军一样精忠报国。"]

    for d in text:
        with torch.no_grad():
            token_ids_test, segment_ids = tokenizer.encode(d, max_length=256)
            token_ids_test = torch.tensor(token_ids_test, device=device).view(1, -1)
            # 先预测subject
            pred_subject = bert_model.predict_subject(token_ids_test)
            pred_subject = pred_subject.squeeze(0)
            subject_texts, subject_idss = search_subject(token_ids_test[0], pred_subject.cpu())
            if len(subject_texts) == 0:
                print("no subject predicted~")
            for sub_text, sub_ids in zip(subject_texts, subject_idss):
                print("subject is " + str(sub_text))
                sub_ids = torch.tensor(sub_ids, device=device).view(1, -1)
                # print("sub_ids shape is " + str(sub_ids))
                object_p_pred = bert_model.predict_object_predicate(token_ids_test, sub_ids)
                res = search_object(token_ids_test[0], object_p_pred.squeeze(0).cpu())
                print("p and obj is " + str(res))



