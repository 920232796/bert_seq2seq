import torch 
from bert_seq2seq import Tokenizer, load_chinese_base_vocab
from bert_seq2seq import load_bert

target = ["O", "B-LOC", "I-LOC", "B-PER", "I-PER", "B-ORG", "I-ORG"]

vocab_path = "./state_dict/roberta_wwm_vocab.txt" # roberta模型字典的位置

model_name = "roberta" # 选择模型名字
model_path = "./state_dict/bert_粗粒度ner_crf.bin"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def viterbi_decode(nodes, trans):
    """
    维特比算法 解码
    nodes: (seq_len, target_size)
    trans: (target_size, target_size)
    """
    with torch.no_grad():
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
                if label[0] == "B":
                    decode.append(label[2: ])
                    starting = True
                elif starting:
                    decode.append(label[2: ])
                else: 
                    starting = False
                    decode.append("O")
            else :
                decode.append("O")
        flag = 0
        res = {}
        for index, each_entity in enumerate(decode):
            if each_entity != "O":
                if flag != each_entity:
                    cur_text = text[index - 1]
                    if each_entity in res.keys():
                        res[each_entity].append(cur_text)
                    else :
                        res[each_entity] = [cur_text]
                    flag = each_entity
                elif flag == each_entity:
                    res[each_entity][-1] += text[index - 1]
            else :
                flag = 0
        print(res)


if __name__ == "__main__":
    vocab_path = "./state_dict/roberta_wwm_vocab.txt"  # roberta模型字典的位置
    model_name = "roberta"  # 选择模型名字
    # 加载字典
    word2idx = load_chinese_base_vocab(vocab_path, simplfied=False)
    tokenizer = Tokenizer(word2idx)
    # 定义模型
    bert_model = load_bert(word2idx, model_name=model_name, model_class="sequence_labeling_crf", target_size=len(target))
    bert_model.set_device(device)
    bert_model.eval()
    ## 加载训练的模型参数～
    bert_model.load_all_params(model_path=model_path, device=device)
    test_data = ["日寇在京掠夺文物详情。", 
                "以书结缘，把欧美，港台流行的食品类食谱汇集一堂。", 
                "明天天津下雨，不知道杨永康主任还能不能来学校吃个饭。",
                "美国的华莱士，我和他谈笑风生",
                "看包公断案的戏"
                ]
    ner_print(bert_model, test_data, device=device)
