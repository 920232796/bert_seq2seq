from re import I
import torch 
import sys
sys.path.append("/Users/xingzhaohu/Downloads/code/python/ml/ml_code/bert/bert_seq2seq")
from bert_seq2seq.utils import load_bert
import torch.nn.functional as F 

def _make_causal_mask(input_ids_shape: torch.Size):
    """
    可以用于cross attention return (tgt_len, tgt_len + past_key_values_len) , row is output, column is input.
    生成一个下三角的mask，lm model 用。
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), 0.0)
    mask_cond = torch.arange(mask.size(-1))
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 1.0)
   
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len)

import os 
input_ids = torch.tensor([[1, 2, 3, 5, -100], [4, 5, 6, -100, -100]])
out = _make_causal_mask(input_ids.shape)
print(out)
print(out.shape)
pad_mask = (input_ids != -100).float()
print(pad_mask.shape)
print(out * pad_mask.unsqueeze(1).unsqueeze(1))
os._exit(0)

# if __name__ == "__main__":
#     t1 = torch.rand(10)
#     print(t1)
#     out = top_k_top_p_filtering(t1, top_k=3)

#     print(out)

#     out = torch.multinomial(F.softmax(out, dim=-1), num_samples=1)

#     print(out)

#     model = load_bert