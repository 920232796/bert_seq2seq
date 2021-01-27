
import sys
sys.path.append("/Users/xingzhaohu/Downloads/code/python/ml/ml_code/bert/bert_seq2seq")
import torch 
import unicodedata
from tqdm import tqdm
import torch.nn as nn 
from torch.optim import Adam
# import pandas as pd
import numpy as np
import os
import json
import time
import bert_seq2seq
from torch.utils.data import Dataset, DataLoader
from bert_seq2seq.tokenizer import Tokenizer, load_chinese_base_vocab
from bert_seq2seq.utils import load_bert
# 共11个标签
target = ["other", "address", "book", "company", "game", "government", "movie", "name", "organization", "position", "scene"]

data_path = "./state_dict/train.json"
vocab_path = "./state_dict/roberta_wwm_vocab.txt" # roberta模型字典的位置

model_name = "roberta" # 选择模型名字
model_path = "./state_dict/roberta_wwm_pytorch_model.bin" # roberta模型位置
recent_model_path = "" # 用于把已经训练好的模型继续训练
model_save_path = "./细粒度_bert_ner_model_crf.bin"
batch_size = 8
lr = 1e-5
crf_lr = 1e-2 ##  crf层学习率为0.01
# 加载字典
word2idx = load_chinese_base_vocab(vocab_path)

def _is_punctuation(ch):
    """标点符号类字符判断（全/半角均在此内）
    """
    code = ord(ch)
    return 33 <= code <= 47 or \
        58 <= code <= 64 or \
        91 <= code <= 96 or \
        123 <= code <= 126 or \
        unicodedata.category(ch).startswith('P')

def _cjk_punctuation():
    return u'\uff02\uff03\uff04\uff05\uff06\uff07\uff08\uff09\uff0a\uff0b\uff0c\uff0d\uff0f\uff1a\uff1b\uff1c\uff1d\uff1e\uff20\uff3b\uff3c\uff3d\uff3e\uff3f\uff40\uff5b\uff5c\uff5d\uff5e\uff5f\uff60\uff62\uff63\uff64\u3000\u3001\u3003\u3008\u3009\u300a\u300b\u300c\u300d\u300e\u300f\u3010\u3011\u3014\u3015\u3016\u3017\u3018\u3019\u301a\u301b\u301c\u301d\u301e\u301f\u3030\u303e\u303f\u2013\u2014\u2018\u2019\u201b\u201c\u201d\u201e\u201f\u2026\u2027\ufe4f\ufe51\ufe54\xb7\uff01\uff1f\uff61\u3002'

def _is_cjk_character(ch):
    """CJK类字符判断（包括中文字符也在此列）
    参考：https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
    """
    code = ord(ch)
    return 0x4E00 <= code <= 0x9FFF or \
        0x3400 <= code <= 0x4DBF or \
        0x20000 <= code <= 0x2A6DF or \
        0x2A700 <= code <= 0x2B73F or \
        0x2B740 <= code <= 0x2B81F or \
        0x2B820 <= code <= 0x2CEAF or \
        0xF900 <= code <= 0xFAFF or \
        0x2F800 <= code <= 0x2FA1F

@staticmethod
def _is_control(ch):
    """控制类字符判断
    """
    return unicodedata.category(ch) in ('Cc', 'Cf')

def word_piece_tokenize(word):
    """word内分成subword
    """
    if word in word2idx:
        return [word]

    tokens = []
    start, stop = 0, 0
    while start < len(word):
        stop = len(word)
        while stop > start:
            sub = word[start:stop]
            if start > 0:
                sub = '##' + sub
            if sub in word2idx:
                break
            stop -= 1
        if start == stop:
            stop += 1
        tokens.append(sub)
        start = stop

    return tokens

def cons_data():
# 按照句号构造输入输出
    # # # 按照句号构造输入输出
    sents = []
    tgts = []
    with open("./state_dict/ner_train_update.txt", "r", encoding="utf-8") as f :
        lines = f.readlines()
        sent = ""
        tgt = []
        for line in lines:
            line = line.strip("\n")
            line = line.split(" ")
            if line[0] == "。":
                sent += line[0]
                tgt.append(line[1])
                if len(sent) < 200:
                    sents.append(sent)
                    tgts.append(tgt)
                sent = ""
                tgt = []
            else :        
                sent += line[0]
                tgt.append(line[1])
    return sents, tgts

## 自定义dataset
class NERDataset(Dataset):
    """
    针对特定数据集，定义一个相关的取数据的方式
    """
    def __init__(self, sents_src, sents_tgt) :
        ## 一般init函数是加载所有数据
        super(NERDataset, self).__init__()
        # 读原始数据
        # self.sents_src, self.sents_tgt = read_corpus(poem_corpus_dir)
        self.sents_src = sents_src
        self.sents_tgt = sents_tgt
       
        self.idx2word = {k: v for v, k in word2idx.items()}
        self.tokenizer = Tokenizer(word2idx)

    def __getitem__(self, i):
        ## 得到单个数据
        # print(i)
        src = self.sents_src[i]
        tgt = self.sents_tgt[i]
        tgt = ["other"] + tgt + ["other"]
        tgt = [target.index(i) for i in tgt ]
        token_ids, token_type_ids = self.tokenizer.encode(src)
        if len(token_ids) != len(tgt):
            print("not equal")
            os._exit(0)
        output = {
            "token_ids": token_ids,
            "token_type_ids": token_type_ids,
            "target_id": tgt
        }
        return output

    def __len__(self):
        return len(self.sents_src)
    
def collate_fn(batch):
    """
    动态padding， batch为一部分sample
    """

    def padding(indice, max_length, pad_idx=0):
        """
        pad 函数
        """
        pad_indice = [item + [pad_idx] * max(0, max_length - len(item)) for item in indice]
        return torch.tensor(pad_indice)

    token_ids = [data["token_ids"] for data in batch]
    
    max_length = max([len(t) for t in token_ids])
    token_type_ids = [data["token_type_ids"] for data in batch]
    target_ids = [data["target_id"] for data in batch]
  
    token_ids_padded = padding(token_ids, max_length)
    token_type_ids_padded = padding(token_type_ids, max_length)
    # target_ids_padded = token_ids_padded[:, 1:].contiguous()
    target_ids_padded = padding(target_ids, max_length)

    return token_ids_padded, token_type_ids_padded, target_ids_padded

def viterbi_decode(nodes, trans):
    """
    维特比算法 解码
    nodes: (seq_len, target_size)
    trans: (target_size, target_size)
    """
    scores = nodes[0]
    scores[1:] -= 100000 # 刚开始标签肯定是"O"
    target_size = nodes.shape[1]
    seq_len = nodes.shape[0]
    labels = torch.arange(0, target_size).view(1, -1)
    path = labels
    for l in range(1, seq_len):
        scores = scores.view(-1, 1)
        M = scores + trans + nodes[l].view(1, -1)
        scores, ids = M.max(0)
        path = torch.cat((path[:, ids], labels), dim=0)
        # print(scores)
    # print(scores)
    return path[:, scores.argmax()]

def ner_print(model, test_data, device="cpu"):
    model.eval()
    idxtword = {v: k for k, v in word2idx.items()}
    tokenier = Tokenizer(word2idx)
    trans = model.state_dict()["crf_layer.trans"]
    for text in test_data:
        decode = []
        text_encode, text_ids = tokenier.encode(text)
        text_tensor = torch.tensor(text_encode, device=device).view(1, -1)
        out = model(text_tensor).squeeze(0) # 其实是nodes
        labels = viterbi_decode(out, trans)
        starting = False
        for l in labels:
            if l > 0:
                label = target[l.item()]
                decode.append(label)
            else :
                decode.append("other")
        flag = 0
        res = {}
        # print(decode)
        # print(text)
        decode_text = [idxtword[i] for i in text_encode]
        for index, each_entity in enumerate(decode):
            if each_entity != "other":
                if flag != each_entity:
                    cur_text = decode_text[index]
                    if each_entity in res.keys():
                        res[each_entity].append(cur_text)
                    else :
                        res[each_entity] = [cur_text]
                    flag = each_entity
                elif flag == each_entity:
                    res[each_entity][-1] += decode_text[index]
            else :
                flag = 0
        print(res)

class Trainer:
    def __init__(self):
        # 加载数据
        sents_src, sents_tgt = cons_data()
        # 判断是否有可用GPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("device: " + str(self.device))
        # 定义模型
        self.bert_model = load_bert(word2idx, model_name=model_name, model_class="sequence_labeling_crf", target_size=len(target))
        ## 加载预训练的模型参数～
        self.bert_model.load_pretrain_params(model_path)
        # 将模型发送到计算设备(GPU或CPU)
        self.bert_model.to(self.device)
        # 声明需要优化的参数
        crf_params = list(map(id, self.bert_model.crf_layer.parameters())) ## 单独把crf层参数拿出来
        base_params = filter(lambda p: id(p) not in crf_params, self.bert_model.parameters())
        self.optimizer = torch.optim.Adam([
                                            {"params": base_params}, 
                                            {"params": self.bert_model.crf_layer.parameters(), "lr": crf_lr}], lr=lr, weight_decay=1e-3)
        # 声明自定义的数据加载器
        dataset = NERDataset(sents_src, sents_tgt)
        self.dataloader =  DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    def train(self, epoch):
        # 一个epoch的训练
        self.bert_model.train()
        self.iteration(epoch, dataloader=self.dataloader, train=True)
    
    def save(self, save_path):
        """
        保存模型
        """
        self.bert_model.save_all_params(save_path)
        print("{} saved!".format(save_path))

    def iteration(self, epoch, dataloader, train=True):
        total_loss = 0
        start_time = time.time() ## 得到当前时间
        step = 0
        for token_ids, token_type_ids, target_ids in tqdm(dataloader,position=0, leave=True):
            # print(target_ids.shape)
            step += 1
            if step % 300 == 0:
                test_data = ["在广州经营小古董珠宝店的潘凝已经收藏了200多款泰迪熊，其中不少更是老牌泰迪熊厂商史蒂夫、赫曼。", 
                "2009年1月，北京市长郭金龙在其政府工作报告中曾明确提出，限价房不停建",
                "昨天，记者连线农业银行亳州市支行办公室主任沈伦，他表示，亳州市支行已经对此事进行了讨论和研究",
                "他们又有会怎样的读书经历。曾经留学海外的香港《号外杂志》主编、著名城市文化学者和作家陈冠中先生"
                ]
                ner_print(self.bert_model, test_data, device=self.device)
                self.bert_model.train()

            token_ids = token_ids.to(self.device)
            token_type_ids = token_type_ids.to(self.device)
            target_ids = target_ids.to(self.device)
            # 因为传入了target标签，因此会计算loss并且返回
            predictions, loss = self.bert_model(token_ids,
                                                labels=target_ids      
                                                )
            # 反向传播
            if train:
                # 清空之前的梯度
                self.optimizer.zero_grad()
                # 反向传播, 获取新的梯度
                loss.backward()
                # 用获取的梯度更新模型参数
                self.optimizer.step()

            # 为计算当前epoch的平均loss
            total_loss += loss.item()
        
        end_time = time.time()
        spend_time = end_time - start_time
        # 打印训练信息
        print("epoch is " + str(epoch)+". loss is " + str(total_loss) + ". spend time is "+ str(spend_time))
        # 保存模型
        self.save(model_save_path)

if __name__ == '__main__':
    
    trainer = Trainer()
    train_epoches = 50
    for epoch in range(train_epoches):
        # 训练一个epoch
        trainer.train(epoch)


    # with open("./state_dict/ner_train.txt", "a+") as f :

    #     for l in open("./state_dict/train.json") :
    #         l = json.loads(l)
            
    #         text = l["text"]
    #         label = l["label"]
    #         for index, char in enumerate(text):
    #             contin = 1
    #             for k, v in label.items():
    #                 if contin == 0:
    #                     break
    #                 for kk, vv in v.items():
    #                     if contin == 0 :
    #                         break
    #                     for each_range in vv:
    #                         if index in range(each_range[0], each_range[1] + 1):
    #                             # 说明存在实体
    #                             f.write(char + " " + k)
    #                             f.write("\n")
    #                             contin = 0
    #                             break
    #             if contin == 1:
    #                 # 说明没找到
    #                 f.write(char + " " + "other")
    #                 f.write("\n")


    # with open("./state_dict/ner_train_update.txt", "a+") as f:
    #     with open("./state_dict/ner_train.txt", "r", encoding="utf-8") as f1 :
    #         lines = f1.readlines()
    #         start = 1
    #         string = ""
    #         label = ""
    #         for line in lines:
    #             line = line.strip("\n")
    #             line = line.split(" ")
    #             if _is_punctuation(line[0]) or _is_cjk_character(line[0]):
    #                 if string != "":
    #                     string = string.lower()
    #                     tokens = word_piece_tokenize(string) # 子词
    #                     for t in tokens:
    #                         if "##" in t:
    #                             f.write(t[2:] + " " + label + "\n")
    #                         else :
    #                             f.write(t + " " + label + "\n")
    #                     # f.write(string + " " + label + "\n")
    #                     string = ""
    #                     label = ""
    #                 f.write(line[0] + " " + line[1] + "\n")
    #             else :
    #                 string += line[0]
    #                 label = line[1]


    # # # 按照句号构造输入输出
    # sents = []
    # tgts = []
    # with open("./state_dict/ner_train_update.txt", "r", encoding="utf-8") as f :
    #     lines = f.readlines()
    #     sent = ""
    #     tgt = []
    #     for line in lines:
    #         line = line.strip("\n")
    #         line = line.split(" ")
    #         if line[0] == "。":
    #             sent += line[0]
    #             tgt.append(line[1])
    #             if len(sent) < 200:
    #                 sents.append(sent)
    #                 tgts.append(tgt)
    #             sent = ""
    #             tgt = []
    #         else :        
    #             sent += line[0]
    #             tgt.append(line[1])
    # index = 0
    
    # tokenizer = Tokenizer(word2idx)
    # for sent, tgt in zip(sents, tgts):
    #     tgt = ["other"] + tgt + ["other"]
    #     tgt = [target.index(i) for i in tgt ]
    #     token_ids, token_type_ids = tokenizer.encode(sent)
    #     if len(token_ids) != len(tgt):
    #         index += 1
    #         print(len(token_ids))
    #         print(len(tgt))
    #         print(tokenizer.decode(token_ids))
    #         print(token_ids)
    #         print(tgt)

    # print(len(sents))
    # print(index)  


