from examples.math_ques_train import load_data
import torch 
import sys
sys.path.append("/Users/xingzhaohu/Downloads/code/python/ml/ml_code/bert/bert_seq2seq")
from bert_seq2seq.utils import load_bert
from bert_seq2seq.utils import top_k_top_p_filtering
import torch.nn.functional as F 

if __name__ == "__main__":
    t1 = torch.rand(10)
    print(t1)
    out = top_k_top_p_filtering(t1, top_k=3)

    print(out)

    out = torch.multinomial(F.softmax(out, dim=-1), num_samples=1)

    print(out)

    model = load_bert