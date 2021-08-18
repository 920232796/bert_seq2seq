"""
"""
import random
import torch
from tqdm import tqdm
import json
from torch.utils.data import Dataset, DataLoader
from bert_seq2seq.tokenizer import Tokenizer, load_chinese_base_vocab
from bert_seq2seq.utils import load_bert
import numpy as np
import time

vocab_path = "./state_dict/roberta_wwm_vocab.txt" # roberta模型字典的位置
model_name = "roberta" # 选择模型名字
model_path = "./state_dict/roberta_wwm_pytorch_model.bin" # roberta模型位置
recent_model_path = ""  # 用于把已经训练好的模型继续训练
model_save_path = "./state_dict/bert_model_relation_extrac.bin"
batch_size = 16
lr = 1e-5

word2idx = load_chinese_base_vocab(vocab_path)
idx2word = {v: k for k, v in word2idx.items()}
tokenizer = Tokenizer(word2idx)

def load_data(filename):
    D = []
    with open(filename, encoding='utf-8') as f:
        for l in f:
            l = json.loads(l)
            D.append({
                'text': l['text'],
                'spo_list': [(spo['subject'], spo['predicate'], spo['object'])
                             for spo in l['spo_list']]
            })
    return D

predicate2id, id2predicate = {}, {}
with open('./state_dict/extract/all_50_schemas') as f:
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


class ExtractDataset(Dataset):
    """
    针对特定数据集，定义一个相关的取数据的方式
    """

    def __init__(self, data):
        ## 一般init函数是加载所有数据
        super(ExtractDataset, self).__init__()
        # 读原始数据
        self.data = data
        self.idx2word = {k: v for v, k in word2idx.items()}

    def __getitem__(self, i):
        ## 得到单个数据
        # print(i)
        d = self.data[i]
        token_ids, segment_ids = tokenizer.encode(d["text"], max_length=256)
        spoes = {}
        for s, p, o in d['spo_list']:
            s = tokenizer.encode(s)[0][1:-1]
            p = predicate2id[p]
            o = tokenizer.encode(o)[0][1:-1]
            s_idx = search(s, token_ids) 
            o_idx = search(o, token_ids)
            if s_idx != -1 and o_idx != -1:
                s = (s_idx, s_idx + len(s) - 1)
                o = (o_idx, o_idx + len(o) - 1, p)
                if s not in spoes:
                    spoes[s] = []
                spoes[s].append(o)
        if spoes:
            # subject标签
            subject_labels = np.zeros((len(token_ids), 2))
            for s in spoes:
                subject_labels[s[0], 0] = 1
                subject_labels[s[1], 1] = 1
            # 随机选一个subject
            start, end = random.choice(list(spoes.keys()))
            subject_ids = (start, end)
            # 对应的object标签
            object_labels = np.zeros((len(token_ids), len(predicate2id), 2))
            for o in spoes.get(subject_ids, []):
                object_labels[o[0], o[2], 0] = 1
                object_labels[o[1], o[2], 1] = 1

            output = {
                "token_ids": token_ids,
                "token_type_ids": segment_ids,
                "subject_labels": subject_labels,
                "subject_ids": subject_ids,
                "object_labels": object_labels,
            }
            return output
        else:
            return self.__getitem__(i + 1)

    def __len__(self):

        return len(self.data)


def collate_fn(batch):
    """
    动态padding， batch为一部分sample
    """

    def padding(inputs, max_length=None, padding=0):
        """Numpy函数，将序列padding到同一长度
        """
        if max_length is None:
            max_length = max([len(x) for x in inputs])

        pad_width = [(0, 0) for _ in np.shape(inputs[0])]
        outputs = []
        for x in inputs:
            x = x[:max_length]
            pad_width[0] = (0, max_length - len(x))
            x = np.pad(x, pad_width, 'constant', constant_values=padding)
            outputs.append(x)

        return np.array(outputs)

    token_ids = [data["token_ids"] for data in batch]
    max_length = max([len(t) for t in token_ids])
    token_type_ids = [data["token_type_ids"] for data in batch]
    subject_labels = [data["subject_labels"] for data in batch]
    object_labels = [data["object_labels"] for data in batch]
    subject_ids = [data["subject_ids"] for data in batch]

    token_ids_padded = padding(token_ids, max_length)
    token_type_ids_padded = padding(token_type_ids, max_length)
    subject_labels_padded = padding(subject_labels, max_length)
    object_labels_padded = padding(object_labels, max_length)
    subject_ids = np.array(subject_ids)

    return torch.tensor(token_ids_padded, dtype=torch.long), torch.tensor(token_type_ids_padded, dtype=torch.float32), \
           torch.tensor(subject_labels_padded, dtype=torch.long), torch.tensor(object_labels_padded, dtype=torch.long), \
           torch.tensor(subject_ids, dtype=torch.long)


class ExtractTrainer:
    def __init__(self):
        # 加载数据
        data_path = "./state_dict/extract/train_data.json"
        data_dev = "./state_dict/extract/dev_data.json"
        self.data = load_data(data_path)
        self.data_dev = load_data(data_dev)

        # 判断是否有可用GPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("device: " + str(self.device))
        # 定义模型
        self.bert_model = load_bert(word2idx, model_name=model_name, model_class="relation_extrac",
                                    target_size=len(predicate2id))
        # 加载预训练的模型参数～
        self.bert_model.load_pretrain_params(model_path)
        # 将模型发送到计算设备(GPU或CPU)
        self.bert_model.set_device(self.device)
        # 声明需要优化的参数
        self.optim_parameters = list(self.bert_model.parameters())
        self.optimizer = torch.optim.Adam(self.optim_parameters, lr=lr, weight_decay=1e-3)
        # 声明自定义的数据加载器
        dataset = ExtractDataset(self.data)
        self.dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        self.best_f1 = 0.0

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

    def test(self, data_dev):
        self.bert_model.eval()
        f = open('./state_dict/dev_pred.json', 'w', encoding='utf-8')
        X, Y, Z = 1e-10, 1e-10, 1e-10
        for tspo in tqdm(data_dev):
            text = tspo["text"]
            spo = tspo["spo_list"]
            token_ids, segment_ids = tokenizer.encode(text, max_length=256)
            token_ids = torch.tensor(token_ids, device=self.device).view(1, -1)
            # 预测 subject
            subject_preds = self.bert_model.predict_subject(token_ids)
            # gpu 写法
            s = np.where(subject_preds.cuda().data.cpu().numpy()[0].T[0] > 0.5)[0]
            e = np.where(subject_preds.cuda().data.cpu().numpy()[0].T[1] > 0.5)[0]
            subject_ix = []
            for i in s:
                end = e[e > i]
                if len(end) > 0:
                    subject_ix.append((i, end[0]))
            # for i,j in subject_ix:
            #     print(tokenizer.decode(token_ids[0][i:j+1].numpy()))
            spoes = []
            for i in subject_ix:
                subject_id = np.array([i])
                object_predicate = self.bert_model.predict_object_predicate(token_ids,
                                                                            torch.tensor(subject_id,device=self.device, dtype=torch.long))
                for object_pred in object_predicate:
                    start = np.where(object_pred.cuda().data.cpu().numpy()[:, :, 0] > 0.5)
                    end = np.where(object_pred.cuda().data.cpu().numpy()[:, :, 1] > 0.5)
                    for _start, predicate1 in zip(*start):
                        for _end, predicate2 in zip(*end):
                            if _start <= _end and predicate1 == predicate2:
                                spoes.append(
                                    (i, predicate1,
                                     (_start, _end))
                                )
                                break
            spoes = [(tokenizer.decode(token_ids.cuda().data.cpu().numpy()[0][i[0]:i[1] + 1]).replace(" ", ""), id2predicate[p],
                      tokenizer.decode(token_ids.cuda().data.cpu().numpy()[0][j[0]:j[1] + 1]).replace(" ", "")) for i, p, j in spoes]

            R = set(spoes)
            T = set(spo)
            X += len(R & T)
            Y += len(R)
            Z += len(T)

            s = json.dumps({
            'text': tspo['text'],
            'spo_list': list(spo),
            'spo_list_pred': list(spoes),
            'new': list(R - T),
            'lack': list(T - R),
            },
                        ensure_ascii=False,
                        indent=4)
            f.write(s + '\n')

        f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
 
        f.close()
        self.bert_model.train()
        return f1, recall, precision

    def iteration(self, epoch, dataloader, train=True):
        total_loss = 0
        start_time = time.time()  # 得到当前时间
        step = 0
        report_loss = 0.0
        last_report_loss = 10000000.0
        for token_ids, token_type_ids, subject_lables, object_labels, subject_ids in tqdm(dataloader):
            step += 1
            if step % 300 == 0:
                print("report loss is " + str(report_loss))
                if report_loss > last_report_loss:
                    self.optimizer.param_groups[0]['lr'] = self.optimizer.param_groups[0]['lr'] / 2
                    print("lr is " + str(self.optimizer.param_groups[0]["lr"]))
                last_report_loss = report_loss
                
                report_loss = 0.0
                text = ["查尔斯·阿兰基斯（Charles Aránguiz），1989年4月17日出生于智利圣地亚哥，智利职业足球运动员，司职中场，效力于德国足球甲级联赛勒沃库森足球俱乐部", 
                "李治即位后，萧淑妃受宠，王皇后为了排挤萧淑妃，答应李治让身在感业寺的武则天续起头发，重新纳入后宫",
                "《星空黑夜传奇》是连载于起点中文网的网络小说，作者是啤酒的罪孽"]
                for d in text:
                    with torch.no_grad():
                        token_ids_test, segment_ids = tokenizer.encode(d, max_length=256)
                        token_ids_test = torch.tensor(token_ids_test, device=self.device).view(1, -1)
                        # 先预测subject
                        pred_subject = self.bert_model.predict_subject(token_ids_test)
                        pred_subject = pred_subject.squeeze(0)
                        subject_texts, subject_idss = search_subject(token_ids_test[0], pred_subject.cpu())
                        if len(subject_texts) == 0:
                            print("no subject predicted~")
                        for sub_text, sub_ids in zip(subject_texts, subject_idss):
                            print("subject is " + str(sub_text))
                            sub_ids = torch.tensor(sub_ids, device=self.device).view(1, -1)
                            # print("sub_ids shape is " + str(sub_ids))
                            object_p_pred = self.bert_model.predict_object_predicate(token_ids_test, sub_ids)
                            res = search_object(token_ids_test[0], object_p_pred.squeeze(0).cpu())
                            print("p and obj is " + str(res))
            if step % 2000 == 0:
                f1, recall, acc = self.test(self.data_dev)
                if f1 > self.best_f1:
                    self.best_f1 = f1
                    # 保存模型
                    self.save(model_save_path)
                print("dev f1: " + str(f1) + " .acc: " + str(acc) + " .recall: " + str(recall) + " best_f1:" + str(self.best_f1))
                  
            # 因为传入了target标签，因此会计算loss并且返回
            predictions, loss = self.bert_model(token_ids,
                                                subject_ids,
                                                subject_labels=subject_lables,
                                                object_labels=object_labels,
    
                                                )
            # 反向传播
            if train:
                # 清空之前的梯度
                self.optimizer.zero_grad()
                # 反向传播, 获取新的梯度
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.bert_model.parameters(), 5.0)
                # 用获取的梯度更新模型参数
                self.optimizer.step()
            # 为计算当前epoch的平均loss
            total_loss += loss.item()
            report_loss += loss.item()

        end_time = time.time()
        spend_time = end_time - start_time
        # 打印训练信息
        print("epoch is " + str(epoch) + ". loss is " + str(total_loss) + ". spend time is " + str(spend_time))

        # f1, recall, acc = self.test(self.data_dev)
        # if f1 > self.best_f1:
        #     self.best_f1 = f1
        #     # 保存模型
        #     self.save(model_save_path)
        # print("dev f1: " + str(f1) + " .acc: " + str(acc) + " .recall: " + str(recall) + " best_f1:" + str(self.best_f1))


if __name__ == "__main__":

    trainer = ExtractTrainer()
    train_epoches = 50
    for epoch in range(train_epoches):
        # 训练一个epoch
        trainer.train(epoch)
