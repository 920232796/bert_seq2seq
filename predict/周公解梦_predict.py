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
    checkpoint = torch.load("./state_dict/bert_dream.model.epoch.16",map_location=torch.device("cpu"))
    ## 加载state dict参数
    bert_seq2seq.load_state_dict(checkpoint)
    bert_seq2seq.eval()

    test_data = ["梦见大街上人群涌动、拥拥而行的景象", "梦见司机将你送到目的地", "梦见别人把弓箭作为礼物送给自己", "梦见中奖了", "梦见大富豪"]
    for text in test_data:
        print(bert_seq2seq.generate(text, beam_size=3))
    
    print(bert_seq2seq.generate("1", beam_size=3))