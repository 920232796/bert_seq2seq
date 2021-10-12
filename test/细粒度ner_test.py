import torch 
from bert_seq2seq import Tokenizer, load_chinese_base_vocab
from bert_seq2seq import load_bert

target = ["other", "address", "book", "company", "game", "government", "movie", "name", "organization", "position", "scene"]

vocab_path = "./state_dict/roberta_wwm_vocab.txt" # roberta模型字典的位置

model_name = "roberta" # 选择模型名字
model_path = "./state_dict/细粒度_bert_ner_model_crf.bin"
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

        return path[:, scores.argmax()]

def ner_print(model, test_data, device="cpu"):
    model.eval()
    idxtword = {v: k for k, v in word2idx.items()}
    tokenier = Tokenizer(word2idx)
    trans = model.state_dict()["crf_layer.trans"]
    for text in test_data:
        decode = []
        text_encode, _ = tokenier.encode(text)
        text_tensor = torch.tensor(text_encode, device=device).view(1, -1)
        out = model(text_tensor).squeeze(0) # 其实是nodes
        labels = viterbi_decode(out, trans)
        starting = False
        for l in labels:
            if l > 0:
                label = target[l.item()]
                decode.append(label)
            else :
                decode.append("other")
        flag = 0
        res = {}
        # print(decode)
        # print(text)
        decode_text = [idxtword[i] for i in text_encode]
        for index, each_entity in enumerate(decode):
            if each_entity != "other":
                if flag != each_entity:
                    # cur_text = "".join([text[t] for t in mapping[index]])

                    cur_text = decode_text[index]
                    if each_entity in res.keys():
                        res[each_entity].append(cur_text)
                    else :
                        res[each_entity] = [cur_text]
                    flag = each_entity
                elif flag == each_entity:
                    res[each_entity][-1] += decode_text[index]
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
    # test_data = ["在广州经营小古董珠宝店的潘凝已经收藏了200多款泰迪熊，其中不少更是老牌泰迪熊厂商史蒂夫、赫曼。", 
    #             "2009年1月，北京市长郭金龙在其政府工作报告中曾明确提出，限价房不停建",
    #             "昨天，记者连线农业银行亳州市支行办公室主任沈伦，他表示，亳州市支行已经对此事进行了讨论和研究",
    #             "他们又有会怎样的读书经历。曾经留学海外的香港《号外杂志》主编、著名城市文化学者和作家陈冠中先生"
    #             ]
    test_data = ["曹操南征荆州，刘表之子刘琮投降，刘备领军民十余万避难，于当阳遭遇曹军追兵，惨败。",
                "赵哲妮（曾用名赵哲），女，1953年11月19日出生，汉族，初中文化程度，户籍所在地河南省漯河市源汇区。"]
    ner_print(bert_model, test_data, device=device)