
import torch 
import numpy as np 
from bert_seq2seq.seq2seq_model import top_k_top_p_filtering
import torch.nn.functional as F 

class ExtendModel:
    def __init__(self, model, tokenizer, bos_id, eos_id, device="cpu") -> None:
        self.model = model 
        self.tokenizer = tokenizer
        self.device = device
        self.bos_id = bos_id
        self.eos_id = eos_id

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def sample_generate_autoregressive(self, text, input_max_length=256, out_max_length=200, top_k=30, top_p=0.0, add_eos=False):

            token_ids = self.tokenizer.encode(text, max_length=input_max_length, truncation=True)
            if add_eos:
                token_ids = token_ids + [self.eos_id]
            token_ids = torch.tensor(token_ids, device=self.device, dtype=torch.long).view(1, -1)
            output_ids = []

            with torch.no_grad():
                for step in range(out_max_length):
                    scores = self.model(input_ids=token_ids)[0]
                    logit_score = torch.log_softmax(scores[:, -1], dim=-1).squeeze(0)
                    if self.tokenizer.unk_token_id is not None:
                        logit_score[self.tokenizer.unk_token_id] = -float('Inf')
                    filtered_logits = top_k_top_p_filtering(logit_score, top_k=top_k, top_p=top_p)
                    next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
                    if self.eos_id == next_token.item():
                        break
                    output_ids.append(next_token.item())
                    token_ids = torch.cat((token_ids, next_token.long().unsqueeze(0)), dim=1)

            return self.tokenizer.decode(output_ids)

    def sample_generate_encoder_decoder(self, text, input_max_length=256, out_max_length=200, top_k=30, top_p=0.0, add_eos=False):


            token_out = self.tokenizer.encode(text, max_length=input_max_length)
            if len(token_out) == 2:
                token_ids = token_out[0]
            else:
                token_ids = token_out
            if add_eos:
                token_ids = token_ids + [self.eos_id]
            token_ids = torch.tensor(token_ids, device=self.device, dtype=torch.long).view(1, -1)
            output_ids = []

            input_decoder_ids = torch.tensor(self.bos_id, device=self.device, dtype=torch.long).view(1, -1)
            with torch.no_grad():
                for step in range(out_max_length):
                    scores = self.model(input_ids=token_ids, decoder_input_ids=input_decoder_ids)[0]
                    logit_score = torch.log_softmax(scores[:, -1], dim=-1).squeeze(0)
                    # if self.tokenizer.unk_token_id is not None:
                    #     logit_score[self.tokenizer.unk_token_id] = -float('Inf')
                    filtered_logits = top_k_top_p_filtering(logit_score, top_k=top_k, top_p=top_p)
                    next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
                    if self.eos_id == next_token.item():
                        break
                        # pass
                    output_ids.append(next_token.item())
                    # token_ids = torch.cat((token_ids, next_token.long().unsqueeze(0)), dim=1)
                    input_decoder_ids = torch.cat((input_decoder_ids, next_token.long().unsqueeze(0)), dim=1)

            return self.tokenizer.decode(output_ids)

    def generate_unilm(self, text, out_max_length=40, beam_size=1, max_length=256):
        # 对 一个 句子生成相应的结果
        self.out_max_length = out_max_length
        input_max_length = max_length - out_max_length
        tokenizer_out = self.tokenizer.encode(text, max_length=input_max_length)
        if len(tokenizer_out) != 1:
            token_ids = tokenizer_out[0]
        else:
            token_ids = tokenizer_out
        token_ids = torch.tensor(token_ids, device=self.device).view(1, -1)
        token_type_ids = torch.zeros_like(token_ids, device=self.device)

        out_puts_ids = self.beam_search(token_ids, token_type_ids, beam_size=beam_size,
                                            device=self.device)

        return self.tokenizer.decode(out_puts_ids.cpu().numpy())

    def beam_search(self, token_ids, token_type_ids, beam_size=1, device="cpu"):
        """
        beam-search操作
        """

        # 用来保存输出序列
        output_ids = torch.empty(1, 0, device=device, dtype=torch.long)
        # 用来保存累计得分

        with torch.no_grad():
            output_scores = torch.zeros(token_ids.shape[0], device=device)
            for step in range(self.out_max_length):
                if step == 0:
                    scores = self.model(token_ids, token_type_ids)
                    # 重复beam-size次 输入ids
                    token_ids = token_ids.view(1, -1).repeat(beam_size, 1)
                    token_type_ids = token_type_ids.view(1, -1).repeat(beam_size, 1)
                else:
                    scores = self.model(new_input_ids, new_token_type_ids)

                logit_score = torch.log_softmax(scores[:, -1], dim=-1)

                logit_score = output_scores.view(-1, 1) + logit_score  # 累计得分
                ## 取topk的时候我们是展平了然后再去调用topk函数
                # 展平
                logit_score = logit_score.view(-1)
                hype_score, hype_pos = torch.topk(logit_score, beam_size)
                indice1 = (hype_pos // scores.shape[-1])  # 行索引
                indice2 = (hype_pos % scores.shape[-1]).long().reshape(-1, 1)  # 列索引

                # 更新得分
                output_scores = hype_score
                output_ids = torch.cat([output_ids[indice1], indice2], dim=1).long()
                new_input_ids = torch.cat([token_ids, output_ids], dim=1)
                new_token_type_ids = torch.cat([token_type_ids, torch.ones_like(output_ids)], dim=1)

                end_counts = (output_ids == self.eos_id).sum(1)  # 统计出现的end标记
                best_one = output_scores.argmax()
                if end_counts[best_one] == 1:
                    # 说明出现终止了～
                    return output_ids[best_one][:-1]
                else:
                    # 保留未完成部分
                    flag = (end_counts < 1)  # 标记未完成序列
                    if not flag.all():  # 如果有已完成的
                        token_ids = token_ids[flag]
                        token_type_ids = token_type_ids[flag]
                        new_input_ids = new_input_ids[flag]
                        new_token_type_ids = new_token_type_ids[flag]
                        output_ids = output_ids[flag]  # 扔掉已完成序列
                        output_scores = output_scores[flag]  # 扔掉已完成序列
                        end_counts = end_counts[flag]  # 扔掉已完成end计数
                        beam_size = flag.sum()  # topk相应变化

            return output_ids[output_scores.argmax()]

