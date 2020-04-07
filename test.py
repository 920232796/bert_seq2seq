import torch 

import torch.nn as nn

from bert_seq2seq.utils import load_bert, load_recent_model
from bert_seq2seq.tokenizer import load_chinese_base_vocab, Tokenizer

target = ["pad", "O", "B-LOC", "I-LOC", "B-PER", "I-PER", "B-ORG", "I-ORG"]

vocab_path = "./state_dict/roberta_wwm_vocab.txt" # roberta模型字典的位置
word2idx = load_chinese_base_vocab(vocab_path)
tokenier = Tokenizer(word2idx)

bert_model = load_bert(vocab_path, model_class="sequence_labeling", target_size=len(target))

bert_model.load_state_dict(torch.load("./state_dict/bert_ner_model.bin", map_location="cpu"))

bert_model.eval()
test_data = ["北京烤鸭好不好吃，只有真的去北京吃过才知道。"]
for text in test_data:
    text_ids, _ = tokenier.encode(text)
    text_ids = torch.tensor(text_ids).view(1, -1)
    out = bert_model(text_ids).squeeze(0)
    out_target = torch.argmax(out, dim=-1)
    decode_target = [target[i.item()] for i in out_target]
    print(decode_target)
    res = {"LOC": [], "ORG": [], "PER": []}
    for index, i in enumerate(decode_target):
        # print(i)
        if i != "O":
            # print(i[-3: ])
            res[i[-3:]].append(text[index - 1])
    
    print(res)

