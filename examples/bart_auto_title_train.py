## model url : https://huggingface.co/fnlp/bart-base-chinese
import torch
import time
import glob
from torch.utils.data import Dataset, DataLoader
from bert_seq2seq import T5PegasusTokenizer, load_chinese_base_vocab
from bert_seq2seq import T5Model
from bert_seq2seq.bart_chinese import BartGenerationModel
from bert_seq2seq import Tokenizer
from tqdm import tqdm
from bert_seq2seq.extend_model_method import ExtendModel

from transformers import BertTokenizer, BartForConditionalGeneration, Text2TextGenerationPipeline

src_dir = './corpus/auto_title/train.src'
tgt_dir = './corpus/auto_title/train.tgt'

vocab_path = "./state_dict/bart-chinese" ## 字典
model_path = "./state_dict/bart-chinese" ## 预训练参数

model_save_path = "./state_dict/bart_autotile.bin" ## 训练完模型 保存在哪里
batch_size = 8
lr = 1e-5

tokenizer = BertTokenizer.from_pretrained(vocab_path)
word2idx = tokenizer.vocab
model = BartForConditionalGeneration.from_pretrained(model_path)

def read_file():
    src = []
    tgt = []

    with open(src_dir,'r',encoding='utf-8') as f:
        lines = f.readlines()

        for line in lines:
            src.append(line.strip('\n').lower())

    with open(tgt_dir,'r',encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            tgt.append(line.strip('\n').lower())

    return src, tgt


class SeqDataset(Dataset):
    """
    针对特定数据集，定义一个相关的取数据的方式
    """

    def __init__(self, sents_src, sents_tgt):
        ## 一般init函数是加载所有数据
        super(SeqDataset, self).__init__()
        # 读原始数据
        self.sents_src = sents_src
        self.sents_tgt = sents_tgt

        self.idx2word = {k: v for v, k in word2idx.items()}

    def __getitem__(self, i):
        ## 得到单个数据
        # print(i)
        src = self.sents_src[i]
        tgt = self.sents_tgt[i]
        token_ids_src = tokenizer.encode(src, max_length=256)
        token_ids_tgt = tokenizer.encode(tgt, max_length=256)
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
    target_ids_padded = target_ids_padded[:, :-1].contiguous()
    labels_ids = labels_ids[:, 1:].contiguous()

    return token_ids_padded, target_ids_padded, labels_ids


class Trainer:
    def __init__(self):
        # 加载数据
        self.sents_src, self.sents_tgt = read_file(src_dir, tgt_dir)

        # 判断是否有可用GPU
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("device: " + str(self.device))
        # 定义模型
        self.model = ExtendModel(model, tokenizer=tokenizer, bos_id=word2idx["[CLS]"], eos_id=word2idx["[SEP]"], device=self.device)

        # 将模型发送到计算设备(GPU或CPU)
        self.model.to(self.device)
        # self.model.set_device(self.device)
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
        for token_ids, target_ids, labels_ids in tqdm(dataloader, total=len(dataloader)):
            step += 1
            token_ids = token_ids.to(self.device)
            target_ids = target_ids.to(self.device)
            labels_ids = labels_ids.to(self.device)
            if step % 100 == 0:
                # self.save(model_save_path)
                self.model.eval()
                test_data = ["本文总结了十个可穿戴产品的设计原则，而这些原则同样也是笔者认为是这个行业最吸引人的地方：1为人们解决重复性问题，2从人开始而不是从机器开始，3要引起注意但不要刻意，4提升用户能力而不是取代人",
                 "2007年乔布斯向人们展示iPhone并宣称它将会改变世界，还有人认为他在夸大其词然而在8年后以iPhone为代表的触屏智能手机已经席卷全球各个角落，未来智能手机将会成为真正的个人电脑为人类发展做出更大的贡献", 
                 "雅虎发布2014年第四季度财报并推出了免税方式剥离其持有的阿里巴巴集团15％股权的计划打算将这一价值约400亿美元的宝贵投资分配给股东截止发稿前雅虎股价上涨了大约7％至5145美元"]
                
                for text in test_data:
                    print(self.model.sample_generate_encoder_decoder(text, add_eos=True, top_k=20))
                self.model.train()
                print("report loss is " + str(report_loss))
                report_loss = 0

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
        # self.save(model_save_path)


if __name__ == '__main__':

    trainer = Trainer()
    train_epoches = 10
    for epoch in range(train_epoches):
        # 训练一个epoch
        trainer.train(epoch)
