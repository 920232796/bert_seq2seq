import torch
from tqdm import tqdm
import time
from torch.utils.data import Dataset, DataLoader
from bert_seq2seq import Tokenizer, load_chinese_base_vocab
from bert_seq2seq import load_bert

data_path = "./corpus/相似句/simtrain_to05sts.txt"
vocab_path = "./state_dict/roberta_wwm_vocab.txt"  # roberta模型字典的位置
word2idx = load_chinese_base_vocab(vocab_path, simplfied=False)
tokenizer = Tokenizer(word2idx)
model_name = "roberta"  # 选择模型名字
model_path = "./state_dict/roberta_wwm_pytorch_model.bin"  # 模型位置

model_save_path = "./state_dict/simbert.bin"
batch_size = 4
lr = 1e-5
maxlen = 256

def read_corpus():
    with open(data_path, encoding="utf-8") as f:
        lines = f.readlines()
    inputs = []
    outputs = []
    for line in lines :
        print(line)
        line = line.split("\t")
        inputs.append(line[1])
        outputs.append(line[3])
    return inputs, outputs

class BertDataset(Dataset):
    """
    针对特定数据集，定义一个相关的取数据的方式
    """

    def __init__(self, inputs, outputs):
        ## 一般init函数是加载所有数据
        super(BertDataset, self).__init__()
        self.inputs = inputs
        self.outputs = outputs
        self.idx2word = {k: v for v, k in word2idx.items()}


    def __getitem__(self, i):
        ## 得到单个数据
        # print(i)
        inp = self.inputs[i]
        out = self.outputs[i]
        token_ids_1, token_type_ids_1 = tokenizer.encode(
            inp, out, max_length=maxlen
        )
        token_ids_2, token_type_ids_2 = tokenizer.encode(
            out, inp, max_length=maxlen
        )

        output = {
            "token_ids_1": token_ids_1,
            "token_type_ids_1": token_type_ids_1,
            "token_ids_2": token_ids_2,
            "token_type_ids_2": token_type_ids_2,

        }
        return output


    def __len__(self):
        return len(self.inputs)


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
    token_ids = []
    token_type_ids = []
    for data in batch:
        token_ids.append(data["token_ids_1"])
        token_type_ids.append(data["token_type_ids_1"])
        token_ids.append(data["token_ids_2"])
        token_type_ids.append(data["token_type_ids_2"])

    max_length = max([len(t) for t in token_ids])

    token_ids_padded = padding(token_ids, max_length)
    token_type_ids_padded = padding(token_type_ids, max_length)
    target_ids_padded = token_ids_padded[:, 1:].contiguous()

    return token_ids_padded, token_type_ids_padded, target_ids_padded


class Trainer:
    def __init__(self):
        # 判断是否有可用GPU
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("device: " + str(self.device))
        # 定义模型
        self.bert_model = load_bert(word2idx, model_name=model_name, model_class="simbert")
        ## 加载预训练的模型参数～
        inputs, outputs = read_corpus()
        self.bert_model.load_pretrain_params(model_path)
        # 加载已经训练好的模型，继续训练

        # 将模型发送到计算设备(GPU或CPU)
        self.bert_model.set_device(self.device)
        # 声明需要优化的参数
        self.optim_parameters = list(self.bert_model.parameters())
        self.optimizer = torch.optim.Adam(self.optim_parameters, lr=lr, weight_decay=1e-3)
        # 声明自定义的数据加载器
        dataset = BertDataset(inputs, outputs)
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
                    "他这个人没个正经的。",
                    "咱俩谁跟谁呀。"
                ]
                for text in test_data:
                    print(self.bert_model.sample_generate(text))
                    print(self.bert_model.sample_generate(text))
                    print(self.bert_model.sample_generate(text))
                    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
                print("loss is " + str(report_loss))
                report_loss = 0
                # self.eval(epoch)
                self.bert_model.train()
            if step % 5000 == 0:
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
    train_epoches = 100

    for epoch in range(train_epoches):
        # 训练一个epoch
        trainer.train(epoch)
