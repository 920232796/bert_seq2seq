
import torch
import glob
import json
import torch
from bert_seq2seq.seq2seq_model import Seq2SeqModel 
# from bert_seq2seq.tokenizer import load_chinese_base_vocab
from transformers import AutoTokenizer, AutoModelForMaskedLM

if __name__ == '__main__':

    # files = glob.glob("./corpus/pdf_full_texts/*.json")

    # index = 0
    # for f in files:
    #     index += 1
    #     with open(f, "r") as ff:
    #         content = ff.read()
    #         content = json.loads(content)
    #         title = content["Title"]
    #         content = content["abstract"]

    #         print(title)
    #         print(content)

    #     if index == 10:
    #         break

    checkpoints = torch.load("./state_dict/bert_english/pytorch_model.bin")

    for k, v in checkpoints.items():
        print(k)

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    word2idx = tokenizer.get_vocab()
    model = Seq2SeqModel(word2idx, model_name="bert", tokenizer = tokenizer)

    for k, v in model.named_parameters():
        if "bert.embeddings.word_embeddings.weight" in k:
            print(k)
            print(v)


    model.load_state_dict(checkpoints, strict=False)
    for k, v in model.named_parameters():
        if "bert.embeddings.word_embeddings.weight" in k:
            print(k)
            print(v)

    t1 = torch.rand(2, 10).long()
    token_type_id = torch.rand(2, 10).long()
    out = model(t1, token_type_id=token_type_id)

    print(out.shape)
