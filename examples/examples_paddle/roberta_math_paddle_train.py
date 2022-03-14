import sys
sys.path.append("/home/bert_seq2seq/paddle_model")
import paddle
import numpy as np
from paddle import nn
from paddlenlp.transformers import AutoTokenizer, AutoModel, BertModel, BertTokenizer, CTRLTokenizer, RobertaTokenizer, \
    RobertaModel
from paddlenlp.transformers.roberta.modeling import robertaSeq2Seq
from paddlenlp.transformers.roberta.modeling import RobertaPretrainedModel
import paddle.nn.functional as F
from tqdm import tqdm
from paddle.io import Dataset
import json
import re

train_data_path = f'./train_data/train.ape.json'
val_data_path = f'./train_data/test.ape.json'


def remove_bucket(equation):
    """去掉冗余的括号
    """
    l_buckets, buckets = [], []
    for i, c in enumerate(equation):
        if c == '(':
            l_buckets.append(i)
        elif c == ')':
            buckets.append((l_buckets.pop(), i))
    eval_equation = eval(equation)
    for l, r in buckets:
        new_equation = '%s %s %s' % (
            equation[:l], equation[l + 1:r], equation[r + 1:]
        )
        try:
            if is_equal(eval(new_equation.replace(' ', '')), eval_equation):
                equation = new_equation
        except:
            pass
    return equation.replace(' ', '')


def is_equal(a, b):
    """比较两个结果是否相等
    """
    a = round(float(a), 6)
    b = round(float(b), 6)
    return a == b


def load_data(filename):
    """读取训练数据，并做一些标准化，保证equation是可以eval的
    参考：https://kexue.fm/archives/7809
    """
    D = []
    # index = 0
    for l in open(filename):
        # index += 1
        # if index == 100:
        #     break
        l = json.loads(l)
        # print(l)
        question, equation, answer = l['original_text'], l['equation'], l['ans']
        # 处理带分数
        question = re.sub('(\d+)\((\d+/\d+)\)', '(\\1+\\2)', question)
        equation = re.sub('(\d+)\((\d+/\d+)\)', '(\\1+\\2)', equation)
        answer = re.sub('(\d+)\((\d+/\d+)\)', '(\\1+\\2)', answer)
        equation = re.sub('(\d+)\(', '\\1+(', equation)
        answer = re.sub('(\d+)\(', '\\1+(', answer)
        # 分数去括号
        question = re.sub('\((\d+/\d+)\)', '\\1', question)
        # 处理百分数
        equation = re.sub('([\.\d]+)%', '(\\1/100)', equation)
        answer = re.sub('([\.\d]+)%', '(\\1/100)', answer)
        # 冒号转除号、剩余百分号处理
        equation = equation.replace(':', '/').replace('%', '/100')
        answer = answer.replace(':', '/').replace('%', '/100')
        if equation[:2] == 'x=':
            equation = equation[2:]
        try:
            # print(equation)
            # print(answer)
            # print("~~~~~~~`")
            if is_equal(eval(equation), eval(answer)):
                D.append((question, remove_bucket(equation), answer))
        except Exception as e:
            print(e)
            continue
    return D


tokenizer = RobertaTokenizer.from_pretrained("roberta-wwm-ext")
# tokenizer = BPETokenizer.from_pretrained("bert-wwm-ext-chinese")

# print(tokenizer)
# text = tokenizer('自然语言处理')
pretrain_model = RobertaModel.from_pretrained("roberta-wwm-ext")
model = robertaSeq2Seq(pretrain_model)
# text_ids = tokenizer.encode("天气好", "北京", max_seq_len=64)
# print(text_ids)



class MyDataset(Dataset):
    """
    步骤一：继承paddle.io.Dataset类
    """

    def __init__(self, data):
        """
        步骤二：实现构造函数，定义数据集大小
        """
        super(MyDataset, self).__init__()
        self.data = data

    def __getitem__(self, i):
        """
        步骤三：实现__getitem__方法，定义指定index时如何获取数据，并返回单条数据（训练数据，对应的标签）
        """

        single_data = self.data[i]
        original_text = single_data[0]
        ans_text = single_data[1]

        text_ids = tokenizer.encode(original_text, ans_text)
        
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
        return len(self.data)

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


data = load_data(train_data_path)
print(data[:5])
single_data = data[0]
original_text = single_data[0]
ans_text = single_data[1]

token_ids, token_type_ids = tokenizer.encode(original_text, ans_text)

print(token_ids)
print(token_type_ids)

dataset = MyDataset(data)
loader = paddle.io.DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=dataset.collate_fn)

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

"""测试代码"""


def eval_equation(equation):
    ans = -10000
    try:
        ans = eval(equation)
    except:
        pass
    return ans 


def validation():
    val_data = load_data(val_data_path)

    right = 0.0
    num = len(val_data)
    # for each_data in tqdm(val_data, total=num):
    for each_data in val_data:

        predictor = Predictor(model, tokenizer, beam_size=2, out_max_length=40, max_length=512)

        equation = predictor.generate(each_data[0])
        
        pred_ans = eval_equation(equation.replace(" ", ""))
        ans1 = each_data[2]
        try :
            if "/" in each_data[2] or "+" in each_data[2] or "-" in each_data[2] or "*" in each_data[2]:
                # print(each_data[2])
                # equation1 = re.sub('\((\d+/\d+)\)', '\\1', str(each_data[2]))
                ans1 = eval(each_data[2])
            if abs(float(pred_ans) - float(ans1)) < 0.01:
                right += 1
                # print("right! pred is " + str(pred_ans) + " ans is " + str(each_data[2]))
            else:
                pass
                # print("err! pred is " + str(pred_ans) + " ans is " + str(each_data[2]))
        except Exception as e:
            print(e)
    

    return right / num 


best_acc = 0


"""测试代码"""








def train():
    paddle.set_device(device="gpu")

    step = 0
    report_loss = 0.0
    for epoch in range(25):
        for inputs, token_type_ids, label in tqdm(loader(), total=len(loader)):
            model.train()
            # print(inputs)
            # print(token_type_ids)
            # print(label)

            step += 1
            if step % 200 == 0:
                predictor = Predictor(model, tokenizer, beam_size=2, out_max_length=40, max_length=512)
                test_data = [
                    "王艳家买了一台洗衣机和一台电冰箱，一共花了6000元，电冰箱的价钱是洗衣机的3/5，求洗衣机的价钱．",
                    "六1班原来男生占总数的2/5，又转来5名男生，现在男生占总数的5/11，女生有多少人？",
                    "两个相同的数相乘，积是3600，这个数是多少."]
                for text in test_data:
                    out = predictor.generate(text)
                    print(f"问题: {text}, 计算式：{out}")
                # print(out)
                print(f"loss is {report_loss}")
                report_loss = 0.0
                pass

            if step % 12500 == 0:
                global best_acc
                print("validing......")
                acc = validation()
                print("valid acc is " + str(acc))
                if acc > best_acc:
                    best_acc = acc
                    ACC = str(best_acc)
                    model.save_pretrained(f'./math/{ACC}/model')
                    tokenizer.save_pretrained(f'./math/{ACC}/tokenizer')
                    saveParams = f'./math/{ACC}/params/bert.pdparams'
                    layer_state_dict = model.state_dict()
                    paddle.save(layer_state_dict, saveParams)
                pass



            loss = model(inputs, token_type_ids=token_type_ids, label=label)
            loss.backward()
            report_loss += loss.item()
            optimizer.step()
            optimizer.clear_grad()

            


def returnForecastForCouplet(data, modelAddress=None, tokenizerAddress=None):
    if modelAddress is None:
        model = robertaSeq2Seq.from_pretrained('./FinallyModelON/')

    else:
        model = robertaSeq2Seq.from_pretrained(modelAddress)

    if tokenizerAddress is None:
        tokenizer = RobertaTokenizer.from_pretrained('./FinallyTokenizerTO/')

    else:
        tokenizer = RobertaTokenizer.from_pretrained(tokenizerAddress)

    predictor = Predictor(model, tokenizer, beam_size=2, out_max_length=40, max_length=512)

    import collections

    OutCouplet = {}

    OutCouplet = collections.OrderedDict()

    for simply_in in data:
        out = predictor.generate(simply_in)
        OutCouplet[simply_in] = out

    return OutCouplet


def forecastForCouplet(data, modelAddress=None, tokenizerAddress=None):
    Couplet = returnForecastForCouplet(data, modelAddress, tokenizerAddress)

    for Uplink, Downlink in Couplet.items():
        print(f"问题：{Uplink}，答案：{Downlink}。")


if __name__ == '__main__':
    
    train()

    model.save_pretrained('./auto/model')
    tokenizer.save_pretrained('./auto/tokenizer')

                

    # predictor = Predictor(model, tokenizer, beam_size=2, out_max_length=10)
    # out = predictor.generate("今天天气好。")
    # print(out)

    # import torch
    # t1 = torch.tensor([[3022]])
    # t2 = torch.tensor([0])
    # print(f"torch 结果：{t1[t2]}")
    #
    # t1 = paddle.to_tensor(np.array([[3022]]))
    # print(t1)
    # t2 = paddle.to_tensor(np.array([0]))
    # print(t2)
    # print(f"paddle 结果：{t1[t2]}")
    #
    # import numpy as np
    # t1 = np.array([[3022]])
    # t2 = np.array([0])
    # print(f"numpy 结果：{t1[t2]}")



