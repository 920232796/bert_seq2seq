import sys
sys.path.append("/home/bert_seq2seq/paddle_model")
import paddle
import numpy as np
from paddle import nn
from paddlenlp.transformers import AutoTokenizer, AutoModel, BertModel, BertTokenizer, CTRLTokenizer,RobertaTokenizer,RobertaModel
from paddlenlp.transformers.roberta.modeling import robertaSeq2Seq
from paddlenlp.transformers.roberta.modeling import RobertaPretrainedModel
import paddle.nn.functional as F
from tqdm import tqdm
from paddle.io import Dataset


tokenizer = RobertaTokenizer.from_pretrained("roberta-wwm-ext")
# tokenizer = BPETokenizer.from_pretrained("bert-wwm-ext-chinese")

# print(tokenizer)
# text = tokenizer('自然语言处理')
pretrain_model = RobertaModel.from_pretrained("roberta-wwm-ext")
model = robertaSeq2Seq(pretrain_model)
# text_ids = tokenizer.encode("天气好", "北京", max_seq_len=64)
# print(text_ids)


data_path = "./corpus/auto_title/"


def read_corpus():
    src_in = []
    src_tgt = []
    with open(data_path + "train.src", mode="r", encoding="utf-8") as f_in:
        lines = f_in.readlines()
    for line in lines:
        src_in.append(line.strip("\n").lower())

    with open(data_path + "train.tgt", mode="r", encoding="utf-8") as f_out:
        lines = f_out.readlines()
    for line in lines:
        src_tgt.append(line.strip("\n").lower())

    return src_in, src_tgt


src_in, src_tgt = read_corpus()

print(src_in[:10])


class MyDataset(Dataset):
    """
    步骤一：继承paddle.io.Dataset类
    """

    def __init__(self):
        """
        步骤二：实现构造函数，定义数据集大小
        """
        super(MyDataset, self).__init__()
        self.src_in = src_in
        self.src_tgt = src_tgt

    def __getitem__(self, index):
        """
        步骤三：实现__getitem__方法，定义指定index时如何获取数据，并返回单条数据（训练数据，对应的标签）
        """
        inputs = self.src_in[index]
        label = self.src_tgt[index]
        text_ids = tokenizer.encode(inputs, label)
        # print(text_ids)
        output = {
            "token_ids": text_ids["input_ids"],
            "token_type_ids": text_ids["token_type_ids"],
        }
        return output

    def __len__(self):
        """
        步骤四：实现__len__方法，返回数据集总数目
        """
        return len(self.src_in)

    @staticmethod
    def collate_fn(batch):
        def padding(indice, max_length, pad_idx=0):
            """
            pad 函数
            """
            pad_indice = [item + [pad_idx] * max(0, max_length - len(item)) for item in indice]
            return paddle.to_tensor(pad_indice)

        token_ids = [data["token_ids"] for data in batch]
        max_length = max([len(t) for t in token_ids])
        token_type_ids = [data["token_type_ids"] for data in batch]

        token_ids_padded = padding(token_ids, max_length)
        token_type_ids_padded = padding(token_type_ids, max_length)
        target_ids_padded = token_ids_padded[:, 1:]

        return token_ids_padded, token_type_ids_padded, target_ids_padded


dataset = MyDataset()
loader = paddle.io.DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=dataset.collate_fn)

optimizer = paddle.optimizer.AdamW(
    learning_rate=1e-5,
    parameters=model.parameters(),
    weight_decay=1e-5)


class Predictor:

    def __init__(self, model: BertModel, tokenizer: BertTokenizer, out_max_length=100, beam_size=1, max_length=512, ):
        self.out_max_length = out_max_length
        self.beam_size = beam_size
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.model = model

    def generate(self, text):
        self.model.eval()
        input_max_length = self.max_length - self.out_max_length
        tokenizer_out = self.tokenizer.encode(text, max_seq_len=input_max_length)
        vocab = self.tokenizer.vocab
        token_ids = tokenizer_out["input_ids"]
        token_type_ids = tokenizer_out["token_type_ids"]
        token_ids = paddle.to_tensor(token_ids).reshape([1, -1])
        token_type_ids = paddle.to_tensor(token_type_ids).reshape([1, -1])

        # print(f"token_ids is {token_ids}")
        out_puts_ids = self.beam_search(token_ids, token_type_ids, vocab, beam_size=self.beam_size)
        # print(out_puts_ids)
        tokens = self.tokenizer.convert_ids_to_tokens(out_puts_ids)

        return self.tokenizer.convert_tokens_to_string(tokens)

    def beam_search(self, token_ids, token_type_ids, word2ix, beam_size=1, ):
        """
        beam-search操作
        """
        sep_id = word2ix["[SEP]"]

        # 用来保存输出序列
        # output_ids = paddle.empty([1, 0]).astype("int")
        output_ids = None
        # output_ids = np.empty([1, 0]).astype(np.int)
        # 用来保存累计得分
        with paddle.no_grad():
            output_scores = np.zeros([token_ids.shape[0]])
            for step in range(self.out_max_length):
                if step == 0:
                    scores = self.model(token_ids, token_type_ids)
                    # 重复beam-size次 输入ids
                    token_ids = np.tile(token_ids.reshape([1, -1]), [beam_size, 1])
                    token_type_ids = np.tile(token_type_ids.reshape([1, -1]), [beam_size, 1])
                else:
                    scores = self.model(new_input_ids, new_token_type_ids)

                logit_score = F.log_softmax(scores[:, -1], axis=-1).numpy()

                logit_score = output_scores.reshape([-1, 1]) + logit_score  # 累计得分
                ## 取topk的时候我们是展平了然后再去调用topk函数
                # 展平
                logit_score = logit_score.reshape([-1])
                hype_pos = np.argpartition(logit_score, -beam_size, axis=-1)[-beam_size:]
                hype_score = logit_score[hype_pos]
                indice1 = (hype_pos // scores.shape[-1]).reshape([-1])  # 行索引
                indice2 = (hype_pos % scores.shape[-1]).astype(np.int).reshape([-1, 1])  # 列索引

                output_scores = hype_score
                if output_ids is None:
                    output_ids = indice2.reshape([beam_size, 1])
                else:
                    output_ids = np.concatenate([output_ids[indice1], indice2], axis=1).astype(np.int)

                new_input_ids = np.concatenate([token_ids, output_ids], axis=1)
                new_token_type_ids = np.concatenate([token_type_ids, np.ones_like(output_ids)], axis=1)

                end_counts = (output_ids == sep_id).sum(1)  # 统计出现的end标记
                best_one = output_scores.argmax()
                if end_counts[best_one] == 1:
                    # 说明出现终止了～
                    return output_ids[best_one][:-1]
                else:
                    # 保留未完成部分
                    flag = (end_counts < 1)  # 标记未完成序列
                    if not flag.all():  # 如果有已完成的
                        token_ids = token_ids[flag]
                        token_type_ids = token_type_ids[flag]
                        new_input_ids = new_input_ids[flag]
                        new_token_type_ids = new_token_type_ids[flag]
                        output_ids = output_ids[flag]  # 扔掉已完成序列
                        output_scores = output_scores[flag]  # 扔掉已完成序列
                        beam_size = flag.sum()  # topk相应变化

            return output_ids[output_scores.argmax()]


def train():
    paddle.set_device(device="gpu")

    step = 0
    report_loss = 0.0
    for epoch in range(10):
        for inputs, token_type_ids, label in tqdm(loader(), total=len(loader)):
            model.train()

            step += 1
            if step % 300 == 0:
                predictor = Predictor(model, tokenizer, beam_size=2, out_max_length=40, max_length=512)
                test_data = [
                    "本文总结了十个可穿戴产品的设计原则而这些原则同样也是笔者认为是这个行业最吸引人的地方1为人们解决重复性问题2从人开始而不是从机器开始3要引起注意但不要刻意4提升用户能力而不是取代人",
                    "2007年乔布斯向人们展示iPhone并宣称它将会改变世界还有人认为他在夸大其词然而在8年后以iPhone为代表的触屏智能手机已经席卷全球各个角落未来智能手机将会成为真正的个人电脑为人类发展做出更大的贡献",
                    "雅虎发布2014年第四季度财报并推出了免税方式剥离其持有的阿里巴巴集团15％股权的计划打算将这一价值约400亿美元的宝贵投资分配给股东截止发稿前雅虎股价上涨了大约7％至5145美元"]
                for text in test_data:
                    out = predictor.generate(text)
                    print(f"新闻: {text}, 标题：{out}")
                print(f"loss is {report_loss}")
                report_loss = 0.0
            loss = model(inputs, token_type_ids=token_type_ids, label=label)
            loss.backward()
            report_loss += loss.item()
            optimizer.step()
            optimizer.clear_grad()


if __name__ == '__main__':
    train()




