
## bert seq2seq 进行英文文本摘要。 模型地址：https://huggingface.co/bert-base-uncased
import torch
from tqdm import tqdm
import torch.nn as nn
from torch.optim import Adam
import numpy as np
import os
import json
import time
import glob
import bert_seq2seq
from torch.utils.data import Dataset, DataLoader
from bert_seq2seq.tokenizer import Tokenizer, load_chinese_base_vocab
from bert_seq2seq.utils import load_bert
from transformers import AutoTokenizer

vocab_path = "./state_dict/roberta_wwm_vocab.txt"  # roberta模型字典的位置
model_name = "roberta"  # 选择模型名字
model_path = "./state_dict/roberta_wwm_pytorch_model.bin"  # 模型位置
recent_model_path = "./state_dict/bert_auto_title_model.bin"  # 用于把已经训练好的模型继续训练
model_save_path = "./state_dict/bert_english_auto_title_model.bin"
batch_size = 4
lr = 1e-5
maxlen = 512
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
word2idx = tokenizer.get_vocab()

def read_data():
    files = glob.glob("./corpus/pdf_full_texts/*.json")

    sents_src = []
    sents_tgt = []
    for f in files:
        with open(f, "r") as ff:
            content = ff.read()
            content = json.loads(content)
            title = content["Title"]
            content = content["abstract"]
            sents_src.append(title)
            sents_tgt.append(content)
    return sents_src, sents_tgt


class BertDataset(Dataset):
    """
    针对特定数据集，定义一个相关的取数据的方式
    """

    def __init__(self):
        ## 一般init函数是加载所有数据
        super(BertDataset, self).__init__()
        ## 拿到所有文件名字
        self.txts = glob.glob("./corpus/pdf_full_texts/*.json")


        self.idx2word = {k: v for v, k in word2idx.items()}

    def __getitem__(self, i):
        ## 得到单个数据
        # print(i)
        text_name = self.txts[i]

        with open(text_name, "r", encoding="utf-8") as f:
            content = f.read()
            content = json.loads(content)
            title = content["Title"]
            content = content["abstract"]

            tokenizer_out = tokenizer.encode_plus(
                content, title, max_length=maxlen,truncation=True
            )
            output = {
                "token_ids": tokenizer_out["input_ids"],
                "token_type_ids": tokenizer_out["token_type_ids"],
            }
            return output


    def __len__(self):
        return len(self.txts)


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

    token_ids_padded = padding(token_ids, max_length)
    token_type_ids_padded = padding(token_type_ids, max_length)
    target_ids_padded = token_ids_padded[:, 1:].contiguous()

    return token_ids_padded, token_type_ids_padded, target_ids_padded


class Trainer:
    def __init__(self):
        # 判断是否有可用GPU
        self.device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
        print("device: " + str(self.device))
        # self.bert_model.load_pretrain_params(model_path, keep_tokens=keep_tokens)
        # 加载已经训练好的模型，继续训练
        checkpoints = torch.load("./state_dict/bert_english/pytorch_model.bin")

        self.bert_model = load_bert(word2idx, tokenizer=tokenizer, model_name="bert")
        self.bert_model.load_state_dict(checkpoints, strict=False)
        # 将模型发送到计算设备(GPU或CPU)
        self.bert_model.set_device(self.device)
        # 声明需要优化的参数
        self.optim_parameters = list(self.bert_model.parameters())
        self.optimizer = torch.optim.Adam(self.optim_parameters, lr=lr, weight_decay=1e-3)
        # 声明自定义的数据加载器
        dataset = BertDataset()
        self.dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

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
        start_time = time.time()  ## 得到当前时间
        step = 0
        report_loss = 0
        for token_ids, token_type_ids, target_ids in tqdm(dataloader, position=0, leave=True):
            step += 1
            if step % 1000 == 0:
                self.bert_model.eval()
                test_data = [
                        "BACKGROUND: On March 20 2020, the Argentine Ministry of Health enforced a mandatory quarantine throughout the country in response to the COVID-19 pandemic. AIMS: The object of this study is to determine the initial impact on mental health of Argentine population, by measuring the prevalence of anxiety, depression, insomnia, and self-perceived stress and by determining the associated risk factors, and to analyze that impact in relation to the number of confirmed cases and deaths. METHOD: A cross-sectional survey was conducted through a digital questionnaire, which was completed by 1,985 respondents between March 29 and April 12. The prevalence of anxiety, depression, stress and insomnia was measured with the Generalized Anxiety Disorder-7 Scale (GAD-7), the 9-Item Patients Health Questionnaire (PHQ-9); the Perceived Stress Scale (PSS-10) and the Pittsburgh Sleep Quality Index (PSQI), respectively. RESULTS: The 62.4% of the surveyed population reported signs of psychological distress. It was found that being a woman, being 18 to 27 years old, living with family members or a partner, smoking, and having a poor sleep quality were the significant risk factors. CONCLUSION: Despite the low number of COVID-19 confirmed cases and deaths at that time, a strong impact on mental health indicators was revealed. The authors of this study recommend the monitoring of the population at risk over time and early interventions in order to avoid long-lasting mental health problems."
                    ]
                for text in test_data:
                    print(self.bert_model.generate(text, beam_size=3, out_max_length=100, max_length=maxlen))
                print("loss is " + str(report_loss))
                report_loss = 0
                # self.eval(epoch)
                self.bert_model.train()
            if step % 8000 == 0:
                self.save(model_save_path)

            # 因为传入了target标签，因此会计算loss并且返回
            predictions, loss = self.bert_model(token_ids,
                                                token_type_ids,
                                                labels=target_ids,

                                                )
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
        # 保存模型
        self.save(model_save_path)


if __name__ == '__main__':

    trainer = Trainer()
    train_epoches = 20

    for epoch in range(train_epoches):
        # 训练一个epoch
        trainer.train(epoch)