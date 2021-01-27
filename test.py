import torch 
from bert_seq2seq.tokenizer import load_chinese_base_vocab
from bert_seq2seq.basic_bert import BertClsClassifier

if __name__ == "__main__":
    word2ix = load_chinese_base_vocab("./state_dict/roberta_wwm_vocab.txt")
    b = BertClsClassifier(word2ix, 14)
    t1 = (torch.rand(1, 5) * 10).long()
    out = b(t1)
    print(out.shape)
    for name, _ in b.named_parameters():
        print(name)

    print("~~~~~~~~~~~~~~~~~~`")

    b.load_pretrain_params("./state_dict/roberta_wwm_pytorch_model.bin")

    b.load_all_params("./state_dict/bert_multi_classify_model.bin", device="cpu")

    