import torch 
import torch.nn as nn 
import numpy as np
from bert_seq2seq.seq2seq_model import top_k_top_p_filtering
from bert_seq2seq.model.gpt2_model import GPT2LMHeadModel, GPT2Config
from bert_seq2seq.basic_bert import BasicGPT
from bert_seq2seq.tokenizer import Tokenizer
import torch.nn.functional as F 

from bert_seq2seq.helper import RepetitionPenaltyLogitsProcessor, TemperatureLogitsProcessor, TopKLogitsProcessor, \
                                TopPLogitsProcessor, ListProcessor

class GPT2(BasicGPT):
    def __init__(self, word2ix, tokenizer=None, 
                   ):
        super().__init__()
        self.word2ix = word2ix
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = Tokenizer(word2ix)
        self.config = GPT2Config(len(word2ix))
        self.model = GPT2LMHeadModel(self.config)
    
    def sample_generate(self, text, input_max_length=256, out_max_length=200, 
                        top_k=30, top_p=1.0, add_eos=False, repetition_penalty=1.0, 
                        temperature=1.0):

        lp = [RepetitionPenaltyLogitsProcessor(penalty=repetition_penalty), 
                TemperatureLogitsProcessor(temperature=temperature),
                TopKLogitsProcessor(top_k=top_k),
                TopPLogitsProcessor(top_p=top_p)
            ]

        self.list_processor = ListProcessor(lp)
        
        token_ids, _ = self.tokenizer.encode(text, max_length=input_max_length)
        if not add_eos:
            token_ids = torch.tensor(token_ids, device=self.device, dtype=torch.long)[:-1].view(1, -1)
        else:
            token_ids = torch.tensor(token_ids, device=self.device, dtype=torch.long).view(1, -1)

        output_ids = []
        sep_id = self.word2ix["[SEP]"]
        with torch.no_grad(): 
            for step in range(out_max_length):
                _, scores = self.model(token_ids)
                logit_score = torch.log_softmax(scores[:, -1], dim=-1)
                logit_score[:, self.word2ix["[UNK]"]] = -float('Inf')

                filtered_logits = self.list_processor(token_ids, logit_score)

                # filtered_logits = top_k_top_p_filtering(logit_score, top_k=top_k, top_p=top_p)
                next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
                if sep_id == next_token.item():
                    break
                output_ids.append(next_token.item())
                token_ids = torch.cat((token_ids, next_token.long()), dim=1)

        return self.tokenizer.decode(np.array(output_ids))

    def sample_generate_once(self, text, input_max_length=256, out_max_length=200, top_k=30, top_p=0.0, sep="。"):
        
        token_ids, _ = self.tokenizer.encode(text, max_length=input_max_length)
        # 不加任何的开始符号和结束符号，就是输入一句话。
        token_ids = torch.tensor(token_ids, device=self.device, dtype=torch.long)[1:-1].view(1, -1)
       

        output_ids = []
        sep_id = self.word2ix[sep] # 句号结尾
        with torch.no_grad(): 
            for step in range(out_max_length):
                _, scores = self.model(token_ids)
                logit_score = torch.log_softmax(scores[:, -1], dim=-1).squeeze(0)
                logit_score[self.word2ix["[UNK]"]] = -float('Inf')
                filtered_logits = top_k_top_p_filtering(logit_score, top_k=top_k, top_p=top_p)
                next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
                if sep_id == next_token.item():
                    break
                output_ids.append(next_token.item())
                token_ids = torch.cat((token_ids, next_token.long().unsqueeze(0)), dim=1)

        return self.tokenizer.decode(np.array(output_ids))

    def sample_generate_english(self, text, input_max_length=256, out_max_length=200, top_k=30, top_p=0.0, add_eos=False):

        token_ids = self.tokenizer.encode(text, max_length=input_max_length, truncation=True)
        if add_eos:
            token_ids = token_ids + [self.word2ix["<EOS>"]]
        token_ids = torch.tensor(token_ids, device=self.device, dtype=torch.long).view(1, -1)
        output_ids = []
        sep_id = self.word2ix["<EOS>"]
        with torch.no_grad():
            for step in range(out_max_length):
                _, scores = self.model(token_ids)
                # print(scores.shape)
                logit_score = torch.log_softmax(scores[:, -1], dim=-1).squeeze(0)
                # print(logit_score.shape)
                logit_score[self.word2ix["unk"]] = -float('Inf')
                filtered_logits = top_k_top_p_filtering(logit_score, top_k=top_k, top_p=top_p)
                next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
                if sep_id == next_token.item():
                    break
                    # pass
                output_ids.append(next_token.item())
                token_ids = torch.cat((token_ids, next_token.long().unsqueeze(0)), dim=1)

        return self.tokenizer.decode(output_ids)


    def _make_causal_mask(self, input_ids_shape: torch.Size):
   
        bsz, tgt_len = input_ids_shape
        mask = torch.full((tgt_len, tgt_len), 0.0).to(self.device)
        mask_cond = torch.arange(mask.size(-1)).to(self.device)
        mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 1.0)
    
        return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len)

    
    def forward(self, x, labels=None):
        if labels is not None:
            labels = labels.to(self.device)
        x = x.to(self.device)
        # input_ids = torch.tensor([[1, 2, 3, 5, -100], [4, 5, 6, -100, -100]])
        attention_mask = self._make_causal_mask(x.shape)
        pad_mask = (labels != -100).float()
        attention_mask = attention_mask * pad_mask.unsqueeze(1).unsqueeze(1)

        loss, lm_logit = self.model(x, labels=labels, attention_mask=attention_mask)
       
        return loss, lm_logit