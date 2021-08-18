import torch
import time
import  glob
from torch.utils.data import Dataset, DataLoader
from bert_seq2seq.tokenizer import T5PegasusTokenizer, load_chinese_base_vocab
from bert_seq2seq.t5_ch import T5Model

vocab_path = "./state_dict/t5-chinese/vocab.txt"
model_path = "./state_dict/t5-chinese/pytorch_model.bin"
model_save_path = "./state_dict/t5_ancient_trans_model.bin"
batch_size = 8
lr = 1e-5
word2idx = load_chinese_base_vocab(vocab_path)
tokenizer = T5PegasusTokenizer(word2idx)


def read_corpus():
    """
    读原始数据
    """
    src = []
    tgt = []
    data_path = glob.glob("./corpus/文言文翻译/*")
    for p in data_path:
        dir = p.split("/")[:-1]
        dir = "/".join(dir)
        # print(dir)
        name = p.split("/")[-1]
        if "翻译" in name:
            # 找到了一个翻译文件
            tgt_name = name
            src_name = name[:-2]
            with open(dir + "/" + src_name) as fs:
                lines = fs.readlines()
                for line in lines:
                    src.append(line.strip("\n").strip())

            with open(dir + "/" + tgt_name) as ft:
                lines = ft.readlines()
                for line in lines:
                    tgt.append(line.strip("\n").strip())

        else:
            pass

    return src, tgt

class SeqDataset(Dataset):
    """
    针对特定数据集，定义一个相关的取数据的方式
    """

    def __init__(self, sents_src, sents_tgt):
        ## 一般init函数是加载所有数据
        super(SeqDataset, self).__init__()
        # 读原始数据
        # self.sents_src, self.sents_tgt = read_corpus(poem_corpus_dir)
        self.sents_src = sents_src
        self.sents_tgt = sents_tgt

        self.idx2word = {k: v for v, k in word2idx.items()}

    def __getitem__(self, i):
        ## 得到单个数据
        # print(i)
        src = self.sents_src[i]
        tgt = self.sents_tgt[i]
        token_ids_src, _ = tokenizer.encode(src, max_length=256)
        token_ids_tgt, _ = tokenizer.encode(tgt, max_length=256)
        output = {
            "token_ids_src": token_ids_src,
            "token_ids_tgt": token_ids_tgt,
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

    token_ids_src = [data["token_ids_src"] for data in batch]
    max_length_src = max([len(t) for t in token_ids_src])
    token_ids_tgt = [data["token_ids_tgt"] for data in batch]
    max_length_tgt = max([len(t) for t in token_ids_tgt])

    token_ids_padded = padding(token_ids_src, max_length_src)
    target_ids_padded = padding(token_ids_tgt, max_length_tgt)
    labels_ids = target_ids_padded.clone()
    labels_ids[labels_ids == 0] = -100
    target_ids_padded = target_ids_padded[:, :-1].contiguous()
    labels_ids = labels_ids[:, 1:].contiguous()

    return token_ids_padded, target_ids_padded, labels_ids


class Trainer:
    def __init__(self):
        # 加载数据
        self.sents_src, self.sents_tgt = read_corpus()

        # 判断是否有可用GPU
        # self.device = torch.device("cpu")
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("device: " + str(self.device))
        # 定义模型
        self.model = T5Model(word2idx)
        ## 加载预训练的模型参数～
        self.model.load_pretrain_params(model_path)
        # 将模型发送到计算设备(GPU或CPU)
        self.model.set_device(self.device)
        # 声明需要优化的参数
        self.optim_parameters = list(self.model.parameters())
        self.optimizer = torch.optim.Adam(self.optim_parameters, lr=lr, weight_decay=1e-3)
        # 声明自定义的数据加载器
        dataset = SeqDataset(self.sents_src, self.sents_tgt)
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
        report_loss = 0
        start_time = time.time()  ## 得到当前时间
        step = 0
        for token_ids, target_ids, labels_ids in dataloader:
            step += 1
            # print(token_ids.shape)
            # print(target_ids.shape)
            # print(labels_ids.shape)
            if step % 4000 == 0:
                self.save(model_save_path)
                self.model.eval()
                test_data = ["遂入颍川。", "会日暝，结陈相持。", "一言兴邦，斯近之矣。"]
                for text in test_data:
                    print(self.model.sample_generate_encoder_decoder(text, add_eos=True))
                self.model.train()
                print("report loss is " + str(report_loss))

            # 因为传入了target标签，因此会计算loss并且返回
            loss = self.model(token_ids,labels=labels_ids, decoder_input_ids=target_ids)[0]
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
            report_loss += loss.item()

        end_time = time.time()
        spend_time = end_time - start_time
        # 打印训练信息
        print("epoch is " + str(epoch) + ". loss is " + str(total_loss) + ". spend time is " + str(spend_time))
        # 保存模型
        self.save(model_save_path)


if __name__ == '__main__':

    trainer = Trainer()
    train_epoches = 10
    for epoch in range(train_epoches):
        # 训练一个epoch
        trainer.train(epoch)
