
import torch
from tqdm import tqdm
import torch.nn as nn
from torch.optim import Adam
import numpy as np
import os
import json
import time
import glob
import pandas as pd
import bert_seq2seq
from torch.utils.data import Dataset, DataLoader
from bert_seq2seq.utils import load_gpt
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("pranavpsv/gpt2-genre-story-generator")
word2ix = tokenizer.get_vocab()
# print(len(word2ix))
# print(word2ix["<EOS>"])
# print(word2ix["<PAD>"])
# print(tokenizer.eos_token_id)

data_path = "./corpus/英文讲故事数据集/train.csv"
model_path = "./state_dict/english_gpt_model/english_gpt_story.bin"
model_save_path = "./state_dict/gpt_auto_story.bin"
batch_size = 8
lr = 1e-5
maxlen = 256

def load_data():
    sents_src = []
    sents_tgt = []
    df = pd.read_csv(data_path)
    for i, row in df.iterrows():
        sents_src.append(row[1])
        tgt = ""
        for j in range(2, 7):
            tgt += row[j]
        sents_tgt.append(tgt)

    return sents_src, sents_tgt

class GPTDataset(Dataset):
    """
    针对特定数据集，定义一个相关的取数据的方式
    """

    def __init__(self):
        ## 一般init函数是加载所有数据
        super(GPTDataset, self).__init__()
        ## 拿到所有文件名字
        self.sents_src, self.sents_tgt = load_data()
        self.tokenizer = tokenizer

    def __getitem__(self, i):
        ## 得到单个数据

        src_d = self.sents_src[i]
        tgt_d = self.sents_tgt[i]
        src_ids = self.tokenizer.encode(src_d) + [self.tokenizer.eos_token_id]
        tgt_ids = self.tokenizer.encode(tgt_d) + [self.tokenizer.eos_token_id]
        output = {
            "token_ids": src_ids + tgt_ids,
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

    token_ids_padded = padding(token_ids, max_length, pad_idx=word2ix["<PAD>"])
    token_target_padded = token_ids_padded.clone()
    token_target_padded[token_target_padded == word2ix["<PAD>"]] = -100
    return token_ids_padded, token_target_padded


class Trainer:
    def __init__(self):
        # 判断是否有可用GPU
        # self.device = torch.device("cpu")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("device: " + str(self.device))
        # 定义模型
        self.model = load_gpt(word2ix, tokenizer=tokenizer)
        self.model.load_pretrain_params(model_path)
        # 加载已经训练好的模型，继续训练
        
        # 将模型发送到计算设备(GPU或CPU)
        self.model.set_device(self.device)
        # 声明需要优化的参数
        self.optim_parameters = list(self.model.parameters())
        self.optimizer = torch.optim.Adam(self.optim_parameters, lr=lr, weight_decay=1e-3)
        # 声明自定义的数据加载器
        dataset = GPTDataset()
        self.dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    def train(self, epoch):
        # 一个epoch的训练
        self.model.train()
        self.iteration(epoch, dataloader=self.dataloader, train=True)

    def save(self, save_path):
        """
        保存模型
        """
        self.model.save_all_params(save_path)
        print("{} saved!".format(save_path))

    def iteration(self, epoch, dataloader, train=True):
        total_loss = 0
        start_time = time.time()  ## 得到当前时间
        step = 0
        report_loss = 0
        for token_ids, token_target in tqdm(dataloader, position=0, leave=True):
            step += 1
            if step % 1000 == 0:
                self.model.eval()
                print(self.model.sample_generate_english("David Drops the Weight", out_max_length=300, add_eos=True))
                print("loss is " + str(report_loss))
                report_loss = 0
                self.model.train()
            if step % 6000 == 0:
                self.save(model_save_path)

            # 因为传入了target标签，因此会计算loss并且返回
            loss, pred_logit = self.model(token_ids, labels=token_target)
            report_loss += loss.item()
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
        print("epoch is " + str(epoch) + ". loss is " + str(total_loss) + ". spend time is " + str(spend_time))


if __name__ == '__main__':

    trainer = Trainer()
    train_epoches = 20

    for epoch in range(train_epoches):
        # 训练一个epoch
        trainer.train(epoch)