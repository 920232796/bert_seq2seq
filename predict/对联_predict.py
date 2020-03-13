import torch
import sys
sys.path.append("/Users/xingzhaohu/Downloads/code/python/ml/ml_code/bert/large_bert")
from model.seq2seq_model import Seq2SeqModel
from model.bert_model import BertConfig
from tokenizer import load_chinese_base_vocab
# model_12 感觉还不错～

if __name__ == "__main__":
    word2idx =load_chinese_base_vocab()
    config = BertConfig(len(word2idx))
    bert_seq2seq = Seq2SeqModel(config)
    ## 加载参数文件
    checkpoint = torch.load("./state_dict/bert_duilian.model.epoch.0",map_location=torch.device("cpu"))
    ## 加载state dict参数
    bert_seq2seq.load_state_dict(checkpoint)
    bert_seq2seq.eval()

    # test_data = [ "国色天香，姹紫嫣红，碧水青云欣共赏", " 一帆风顺年年好", "落花因蝶舞", "三千世界笙歌里", "前程似锦"]
    # for text in test_data:
    #     print(bert_seq2seq.generate(text, beam_size=3))
    
    # print(bert_seq2seq.generate("前程似锦", beam_size=3))

    while (True):
        in_str = input("请输入上联:")
        if (in_str == "q"):
            print("bye~")
            break 
        print(bert_seq2seq.generate(in_str.strip()))