## 自动写诗的例子
import sys
sys.path.append("/Users/xingzhaohu/Downloads/code/python/ml/ml_code/bert/bert_seq2seq")
import torch 
from tqdm import tqdm
import torch.nn as nn 
from torch.optim import Adam
import pandas as pd
import numpy as np
import os
import json
import time
from torch.utils.data import Dataset, DataLoader
from bert_seq2seq.tokenizer import Tokenizer, load_chinese_base_vocab
from bert_seq2seq.utils import load_bert, load_model_params, load_recent_model
# 引入自定义数据集
from bert_seq2seq.bert_dataset import BertDataset

def read_corpus(dir_path, vocab_path):
    """
    读原始数据
    """
    sents_src = []
    sents_tgt = []
    word2idx = load_chinese_base_vocab(vocab_path)
    tokenizer = Tokenizer(word2idx)
    files= os.listdir(dir_path) #得到文件夹下的所有文件名称
    for file1 in files: #遍历文件夹
        if not os.path.isdir(file1): #判断是否是文件夹，不是文件夹才打开
            file_path = dir_path + "/" + file1
            print(file_path)
            if file_path[-3:] != "csv":
                continue
            df = pd.read_csv(file_path)
            # 先判断诗句的类型  再确定是否要构造数据
            for index, row in df.iterrows():
                if type(row[0]) is not str or type(row[3]) is not str:
                    continue
                if len(row[0]) > 8 or len(row[0]) < 2:
                    # 过滤掉题目长度过长和过短的诗句
                    continue
                if len(row[0].split(" ")) > 1:
                    # 说明题目里面存在空格，只要空格前面的数据
                    row[0] = row[0].split(" ")[0]
                encode_text = tokenizer.encode(row[3])[0]
                if word2idx["[UNK]"] in encode_text:
                  # 过滤unk字符
                  continue
                if len(row[3]) == 24 and (row[3][5] == "，" or row[3][5] == "。"):
                    # 五言绝句
                    sents_src.append(row[0] + "##" + "五言绝句")
                    sents_tgt.append(row[3])
                elif len(row[3]) == 32 and (row[3][7] == "，" or row[3][7] == "。"):
                    # 七言绝句
                    sents_src.append(row[0] + "##" + "七言绝句")
                    sents_tgt.append(row[3])
                elif len(row[3]) == 48 and (row[3][5] == "，" or row[3][5] == "。"):
                    # 五言律诗
                    sents_src.append(row[0] + "##" + "五言律诗")
                    sents_tgt.append(row[3])
                elif len(row[3]) == 64 and (row[3][7] == "，" or row[3][7] == "。"):
                    # 七言律诗
                    sents_src.append(row[0] + "##" + "七言律诗")
                    sents_tgt.append(row[3])

    print("诗句共: " + str(len(sents_src)) + "篇")
    return sents_src, sents_tgt

def collate_fn(batch):
    """
    动态padding， batch为一部分sample
    """

    def padding(indice, max_length, pad_idx=0):
        """
        pad 函数
        注意 token type id 右侧pad是添加1而不是0，1表示属于句子B
        """
        pad_indice = [item + [pad_idx] * max(0, max_length - len(item)) for item in indice]
        return torch.tensor(pad_indice)

    token_ids = [data["token_ids"] for data in batch]
    max_length = max([len(t) for t in token_ids])
    token_type_ids = [data["token_type_ids"] for data in batch]

    token_ids_padded = padding(token_ids, max_length)
    token_type_ids_padded = padding(token_type_ids, max_length)
    target_ids_padded = token_ids_padded[:, 1:].contiguous()

    return token_ids_padded, token_type_ids_padded, target_ids_padded

class PoemTrainer:
    def __init__(self):
        # 加载数据
        data_dir = "./corpus/Poetry"
        self.vocab_path = "./state_dict/roberta_wwm_vocab.txt" # roberta模型字典的位置
        self.sents_src, self.sents_tgt = read_corpus(data_dir, self.vocab_path)
        self.model_name = "roberta" # 选择模型名字
        self.model_path = "./state_dict/roberta_wwm_pytorch_model.bin" # roberta模型位置
        self.recent_model_path = "" # 用于把已经训练好的模型继续训练
        self.model_save_path = "./bert_model.bin"
        self.batch_size = 16
        self.lr = 1e-5
        # 加载字典
        self.word2idx = load_chinese_base_vocab(self.vocab_path)
        # 判断是否有可用GPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("device: " + str(self.device))
        # 定义模型
        self.bert_model = load_bert(self.vocab_path, model_name=self.model_name)
        ## 加载预训练的模型参数～
        load_model_params(self.bert_model, self.model_path)
        # 将模型发送到计算设备(GPU或CPU)
        self.bert_model.to(self.device)
        # 声明需要优化的参数
        self.optim_parameters = list(self.bert_model.parameters())
        self.optimizer = torch.optim.Adam(self.optim_parameters, lr=self.lr, weight_decay=1e-3)
        # 声明自定义的数据加载器
        dataset = BertDataset(self.sents_src, self.sents_tgt, self.vocab_path)
        self.dataloader =  DataLoader(dataset, batch_size=self.batch_size, shuffle=True, collate_fn=collate_fn)

    def train(self, epoch):
        # 一个epoch的训练
        self.bert_model.train()
        self.iteration(epoch, dataloader=self.dataloader, train=True)

    def iteration(self, epoch, dataloader, train=True):
        total_loss = 0
        start_time = time.time() ## 得到当前时间
        step = 0
        for token_ids, token_type_ids, target_ids in tqdm(dataloader,position=0, leave=True):
            step += 1
            if step % 3000 == 0:
                self.bert_model.eval()
                test_data = ["观棋##五言绝句", "题西林壁##七言绝句", "长安早春##五言律诗"]
                for text in test_data:
                    print(self.bert_model.generate(text, beam_size=3,device=self.device, is_poem=True))
                self.bert_model.train()

            token_ids = token_ids.to(self.device)
            token_type_ids = token_type_ids.to(self.device)
            target_ids = target_ids.to(self.device)
            # 因为传入了target标签，因此会计算loss并且返回
            predictions, loss = self.bert_model(token_ids,
                                                token_type_ids,
                                                labels=target_ids,
                                                device=self.device
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
        self.bert_model.save(self.model_save_path)

if __name__ == '__main__':

    # word2idx = load_chinese_base_vocab()
    # tokenier = Tokenizer(word2idx)

    trainer = PoemTrainer()
    train_epoches = 10
    for epoch in range(train_epoches):
        # 训练一个epoch
        trainer.train(epoch)

    ## 测试一下read corpus
    # sents_src, sents_tgt = read_corpus(duilian_corpus_dir)
    # print(sents_src[:5])
    # print(sents_tgt[:5])
    # print(tokenier.encode(sents_src[0]))

    # # 测试一下自定义数据集
    # vocab_path = "./state_dict/roberta_wwm_vocab.txt" # roberta模型字典的位置
    # sents_src, sents_tgt = read_corpus("./corpus/Poetry", vocab_path)
    
    # dataset = BertDataset(sents_src, sents_tgt, vocab_path)
    # word2idx = load_chinese_base_vocab(vocab_path)
    # tokenier = Tokenizer(word2idx)
    # dataloader =  DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
    # for token_ids, token_type_ids, target_ids in dataloader:
    #     print(token_ids.shape)
    #     print(tokenier.decode(token_ids[0].tolist()))
    #     print(tokenier.decode(token_ids[1].tolist()))
    #     print(token_type_ids)
    #     print(target_ids.shape)
    #     print(tokenier.decode(target_ids[0].tolist()))
    #     print(tokenier.decode(target_ids[1].tolist()))
    #     break


    # 
    # read_corpus(poem_corpus_dir)
    