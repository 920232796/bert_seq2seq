import sys
sys.path.append("/Users/xingzhaohu/Downloads/code/python/ml/ml_code/bert/bert_seq2seq")

import torch
from bert_seq2seq.utils import load_gpt
from bert_seq2seq.tokenizer import load_chinese_base_vocab
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

import os
from bert_seq2seq.extend_model_method import ExtendModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_path = "./state_dict/gpt_auto_story.bin"

if __name__ == "__main__":

    tokenizer = AutoTokenizer.from_pretrained("/Users/xingzhaohu/Downloads/t5_test")
    model = AutoModelForSeq2SeqLM.from_pretrained("/Users/xingzhaohu/Downloads/t5_test")
    model.eval()
    model.to(device)
    model = ExtendModel(model, tokenizer, bos_id=0, eos_id=1)
    print(model.sample_generate_encoder_decoder("translate English to German: That is good", out_max_length=300, add_eos=True))


    t1 = (torch.rand(2, 10) * 10).long()
    label = (torch.rand(2, 10) * 10).long()
    # (2, 10)
    out = model(input_ids=t1, decoder_input_ids=label, labels=label)
    print(len(out))
    print(out[0])


    # # decoder_input = label[:, :-1]
    # # labels = label[:, 1:]
    #
    # out = model(input_ids=t1, decoder_input_ids=label, labels=label)
    # print(out[0])
    #
    # print(out[1].shape)


