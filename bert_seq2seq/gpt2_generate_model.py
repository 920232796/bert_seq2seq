import torch 
import torch.nn as nn 
import numpy as np
from bert_seq2seq.seq2seq_model import top_k_top_p_filtering
from bert_seq2seq.model.gpt2_model import GPT2LMHeadModel, GPT2Config
from bert_seq2seq.basic_bert import BasicGPT
from bert_seq2seq.tokenizer import Tokenizer
import torch.nn.functional as F 

class GPT2(BasicGPT):
    def __init__(self, word2ix):
        super().__init__()
        self.word2ix = word2ix
        self.tokenizer = Tokenizer(word2ix)
        self.config = GPT2Config(len(word2ix))
        self.model = GPT2LMHeadModel(self.config)
    
    def sample_generate(self, text, input_max_length=256, out_max_length=200, top_k=30, top_p=0.0):
        
        token_ids, _ = self.tokenizer.encode(text, max_length=input_max_length)
        
        token_ids = torch.tensor(token_ids, device=self.device, dtype=torch.long)[:-1].view(1, -1)
        output_ids = []
        sep_id = self.word2ix["[SEP]"]
        with torch.no_grad(): 
            for step in range(out_max_length):
                _, scores = self.model(token_ids)
                # print(scores.shape)
                logit_score = torch.log_softmax(scores[:, -1], dim=-1).squeeze(0)
                # print(logit_score.shape)
                logit_score[self.word2ix["[UNK]"]] = -float('Inf')
                filtered_logits = top_k_top_p_filtering(logit_score, top_k=top_k, top_p=top_p)
                next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
                if sep_id == next_token.item():
                    break
                output_ids.append(next_token.item())
                token_ids = torch.cat((token_ids, next_token.long().unsqueeze(0)), dim=1)

        return self.tokenizer.decode(np.array(output_ids))


    def _make_causal_mask(self, input_ids_shape: torch.Size):
   
        bsz, tgt_len = input_ids_shape
        mask = torch.full((tgt_len, tgt_len), 0.0).to(self.device)
        mask_cond = torch.arange(mask.size(-1)).to(self.device)
        mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 1.0)
    
        return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len)

    
    def forward(self, x, label=None):

        # input_ids = torch.tensor([[1, 2, 3, 5, -100], [4, 5, 6, -100, -100]])
        attention_mask = self._make_causal_mask(x.shape)
        pad_mask = (x != -100).float()
        attention_mask = attention_mask * pad_mask.unsqueeze(1).unsqueeze(1)

        loss, lm_logit = self.model(x, label, attention_mask=attention_mask)
       
        return loss, lm_logit