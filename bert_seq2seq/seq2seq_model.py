import torch 
import torch.nn as nn 
import torch.nn.functional as F
import random
from bert_seq2seq.tokenizer import Tokenizer, load_chinese_base_vocab
import time
from bert_seq2seq.config import yayun_list
import os 
from bert_seq2seq.basic_bert import BasicBert
import numpy as np
from bert_seq2seq.helper import RepetitionPenaltyLogitsProcessor, TemperatureLogitsProcessor, TopKLogitsProcessor, \
                                TopPLogitsProcessor, ListProcessor


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits

class Seq2SeqModel(BasicBert):
    """
    """
    def __init__(self, word2ix, model_name="roberta", tokenizer=None):
        super(Seq2SeqModel, self).__init__(word2ix=word2ix, model_name=model_name, tokenizer=tokenizer)
            
        self.hidden_dim = self.config.hidden_size
        self.vocab_size = len(word2ix)

    def compute_loss(self, predictions, labels, target_mask):
        """
        target_mask : 句子a部分和pad部分全为0， 而句子b部分为1
        """
        predictions = predictions.view(-1, self.vocab_size)
        labels = labels.view(-1)
        target_mask = target_mask.view(-1).float()
        loss = nn.CrossEntropyLoss(ignore_index=0, reduction="none")
        return (loss(predictions, labels) * target_mask).sum() / target_mask.sum() ## 通过mask 取消 pad 和句子a部分预测的影响
    
    def forward(self, input_tensor, token_type_id, position_enc=None, labels=None):
        input_tensor = input_tensor.to(self.device)
        token_type_id = token_type_id.to(self.device)
        if position_enc is not None:
            position_enc = position_enc.to(self.device)
        if labels is not None :
            labels = labels.to(self.device)
        input_shape = input_tensor.shape
        batch_size = input_shape[0]
        seq_len = input_shape[1]
        ## 构建特殊的mask
        ones = torch.ones((1, 1, seq_len, seq_len), dtype=torch.float32, device=self.device)
        a_mask = ones.tril()
        s_ex12 = token_type_id.unsqueeze(1).unsqueeze(2).float()
        s_ex13 = token_type_id.unsqueeze(1).unsqueeze(3).float()
        a_mask = (1.0 - s_ex12) * (1.0 - s_ex13) + s_ex13 * a_mask
            
        enc_layers, _ = self.bert(input_tensor, position_ids=position_enc, token_type_ids=token_type_id, attention_mask=a_mask, 
                                    output_all_encoded_layers=True)
        squence_out = enc_layers[-1] ## 取出来最后一层输出 (batch, seq_len, 768)

        tokens_hidden_state, predictions = self.cls(squence_out)

        if labels is not None:

            predictions = predictions[:, :-1].contiguous()
            target_mask = token_type_id[:, 1:].contiguous()
            loss = self.compute_loss(predictions, labels, target_mask)
            return predictions, loss 
        else :
            return predictions

    
    def generate(self, text, out_max_length=40, beam_size=1, is_poem=False, max_length=256):

        self.out_max_length = out_max_length
        input_max_length = max_length - out_max_length
        # print(text)
        try:
            token_ids, token_type_ids = self.tokenizer.encode(text, max_length=input_max_length)
        except:
            # 可能是transformer的tokenizer
            tokenizer_out = self.tokenizer.encode_plus(text, max_length=input_max_length, truncation=True)
            token_ids = tokenizer_out["input_ids"]
            token_type_ids = tokenizer_out["token_type_ids"]
        token_ids = torch.tensor(token_ids, device=self.device).view(1, -1)
        token_type_ids = torch.tensor(token_type_ids, device=self.device).view(1, -1)
        if is_poem:## 古诗的beam-search稍有不同
            
            out_puts_ids = self.beam_search_poem(text, token_ids, token_type_ids, self.word2ix, beam_size=beam_size, device=self.device)
        else :   
            out_puts_ids = self.beam_search(token_ids, token_type_ids, self.word2ix, beam_size=beam_size, device=self.device)
        
        return self.tokenizer.decode(out_puts_ids.cpu().numpy())
    
    def generate_random(self, text, out_max_length=40, beam_size=3, max_length=256):
        # 对 一个 句子生成相应的结果
        ## 通过输出最大长度得到输入的最大长度，这里问题不大，如果超过最大长度会进行截断
        self.out_max_length = out_max_length
        input_max_length = max_length - out_max_length
        token_ids, token_type_ids = self.tokenizer.encode(text, max_length=input_max_length)
        token_ids = torch.tensor(token_ids, device=self.device).view(1, -1)
        token_type_ids = torch.tensor(token_type_ids, device=self.device).view(1, -1)
         
        out_puts_ids_list = self.beam_search_list(token_ids, token_type_ids, self.word2ix, beam_size=beam_size, device=self.device)
        random_int = random.randint(0, len(out_puts_ids_list) - 1)
        return self.tokenizer.decode(out_puts_ids_list[random_int].cpu().numpy())

    def sample_generate(self, text, out_max_length=40, top_k=30, 
                            top_p=0.0, max_length=256, repetition_penalty=1.0, 
                        temperature=1.0):

        input_max_length = max_length - out_max_length
        token_ids, token_type_ids = self.tokenizer.encode(text, max_length=input_max_length)


        lp = [RepetitionPenaltyLogitsProcessor(penalty=repetition_penalty), 
                TemperatureLogitsProcessor(temperature=temperature),
                TopKLogitsProcessor(top_k=top_k),
                TopPLogitsProcessor(top_p=top_p),
            ]
        list_processor = ListProcessor(lp) 

        token_ids = torch.tensor(token_ids, device=self.device, dtype=torch.long).view(1, -1)
        token_type_ids = torch.tensor(token_type_ids, device=self.device, dtype=torch.long).view(1, -1)
        device = self.device
        output_ids = []
        sep_id = self.word2ix["[SEP]"]
        with torch.no_grad(): 
            for step in range(out_max_length):
                scores = self.forward(token_ids, token_type_ids)
                logit_score = torch.log_softmax(scores[:, -1], dim=-1)
                logit_score[:, self.word2ix["[UNK]"]] = -float('Inf')

                filtered_logits = list_processor(token_ids, logit_score)
                
                # filtered_logits = top_k_top_p_filtering(logit_score, top_k=top_k, top_p=top_p)
                next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
                if sep_id == next_token.item():
                    break
                output_ids.append(next_token.item())
                token_ids = torch.cat((token_ids, next_token.long()), dim=1)
                token_type_ids = torch.cat([token_type_ids, torch.ones((1, 1), device=device, dtype=torch.long)], dim=1)

        return self.tokenizer.decode(np.array(output_ids))

    
    # def poem_beam_search(self, token_ids, token_type_ids, word2ix, beam_size=1, device="cpu"):
    #     """
    #     专门针对写诗的beam-search
    #     """
    #     ix2word = {v: k for k, v in word2ix.items()}
    #     sep_id = word2ix["[SEP]"]
    #     douhao_id = word2ix["，"]# 逗号
    #     juhao_id = word2ix["。"]# 句号
    #     # 用来保存输出序列
    #     output_ids = [[]]
    #     # word_list = {} # 保证不重复生成
    #     repeat_list = [[], [], [], [], []]
    #     last_chars = []
    #     yayun_save = -1
    #     # 用来保存累计得分
    #     output_scores = torch.zeros(token_ids.shape[0], device=device)
    #     flag = 0 # 判断第一次遇到逗号
    #     for step in range(self.out_max_length):
    #         scores = self.forward(token_ids, token_type_ids, device=device)
    #         if step == 0:
    #             # 重复beam-size次 输入ids
    #             token_ids = token_ids.view(1, -1).repeat(beam_size, 1)
    #             token_type_ids = token_type_ids.view(1, -1).repeat(beam_size, 1)
    #         ## 计算log 分值 (beam_size, vocab_size)
    #         logit_score = torch.log_softmax(scores, dim=-1)[:, -1]
    #         logit_score = output_scores.view(-1, 1) + logit_score # 累计得分
    #         ## 取topk的时候我们是展平了然后再去调用topk函数
    #         # 展平
    #         logit_score = logit_score.view(-1)
    #         hype_score, hype_pos = torch.topk(logit_score, beam_size)
    #         indice1 = hype_pos // scores.shape[-1] # 行索引
    #         indice2 = hype_pos % scores.shape[-1] # 列索引

    #         # 下面需要更新一下输出了
    #         new_hype_scores = []
    #         new_hype_ids = []
    #         new_repeat_list = []
    #         next_chars = [] # 用来保存新预测出来的一个字符，继续接到输入序列后面，再去预测新字符
    #         index = 0
    #         for i_1, i_2, score in zip(indice1, indice2, hype_score):
    #             i_1 = i_1.item()
    #             i_2 = i_2.item()
    #             score = score.item()
    #             if i_2 != douhao_id and i_2 != juhao_id:
    #                 if i_2 in repeat_list[i_1]:
    #                 # 说明出现重复了
    #                 # 扣分
    #                     score -= 1
    #                     hype_score[i_1] -= 1
    #                 else :
    #                     repeat_list[i_1].append(i_2)

    #                 # if i_2 not in word_list.keys():
    #                 #     word_list[i_2] = 1
    #                 # else :
    #                 #     # 加惩罚
    #                 #     word_list[i_2] += 1
    #                 #     score -= 1 * word_list[i_2]
    #                 #     hype_score[index] -= 1 * word_list[i_2]
    #             if flag == 0 and i_2 == douhao_id and len(last_chars) != 0:
                    
    #                 flag += 1
    #                 word = ix2word[last_chars[index]]# 找到上一个字符 记住其押韵情况
    #                 for i, each_yayun in enumerate(yayun_list):
    #                     if word in each_yayun:
    #                         yayun_save = i
    #                         break
    #             if i_2 == juhao_id and len(last_chars) != 0:
                    
    #                 word = ix2word[last_chars[i_1]]
    #                 # 找押韵 给奖励
    #                 if word in yayun_list[yayun_save]:
    #                     score += 2
    #                     hype_score[i_1] += 2
    #                 else:
    #                     score -= 2
    #                     hype_score[i_1] -= 2
    #             hype_id = output_ids[i_1] + [i_2] # 保存所有输出的序列，而不仅仅是新预测的单个字符

    #             if i_2 == sep_id:
    #                 # 说明解码到最后了
    #                 if score == torch.max(hype_score).item():
    #                     return hype_id[: -1], False
    #                 else:
    #                     # 完成一个解码了，但这个解码得分并不是最高，因此的话需要跳过这个序列
    #                     beam_size -= 1
    #             else :
    #                 new_hype_ids.append(hype_id)
    #                 new_hype_scores.append(score)
    #                 next_chars.append(i_2) # 收集一下，需要连接到当前的输入序列之后
    #                 new_repeat_list.append(repeat_list[i_1])
    #             index += 1

    #         output_ids = new_hype_ids
    #         repeat_list = new_repeat_list ## 重复会扣分
    #         last_chars = next_chars.copy() # 记录一下上一个字符
    #         output_scores = torch.tensor(new_hype_scores, dtype=torch.float32, device=device)
    #         # 现在需要重新构造输入数据了，用上一次输入连接上这次新输出的字符，再输入bert中预测新字符
    #         token_ids = token_ids[:len(output_ids)].contiguous() # 截取，因为要过滤掉已经完成预测的序列
    #         token_type_ids = token_type_ids[: len(output_ids)].contiguous()

    #         next_chars = torch.tensor(next_chars, dtype=torch.long, device=device).view(-1, 1)
    #         next_token_type_ids = torch.ones_like(next_chars, device=device)
    #         # 连接
    #         token_ids = torch.cat((token_ids, next_chars), dim=1)
    #         token_type_ids = torch.cat((token_type_ids, next_token_type_ids), dim=1)
    #         if beam_size < 1:
    #             break

    #     # 如果达到最大长度的话 直接把得分最高的输出序列返回把
    #     err = False
    #     try: 
    #         return output_ids[output_scores.argmax().item()], err
    #     except:
    #         err = True
    #         return "本次解码出现错误", err

    def beam_search(self, token_ids, token_type_ids, word2ix, beam_size=1, device="cpu"):
        """
        beam-search操作
        """
        sep_id = word2ix["[SEP]"]
        
        # 用来保存输出序列
        output_ids = torch.empty(1, 0, device=device, dtype=torch.long)
        # 用来保存累计得分
      
        with torch.no_grad(): 
            output_scores = torch.zeros(token_ids.shape[0], device=device)
            for step in range(self.out_max_length):
                if step == 0:
                    scores = self.forward(token_ids, token_type_ids)
                    # 重复beam-size次 输入ids
                    token_ids = token_ids.view(1, -1).repeat(beam_size, 1)
                    token_type_ids = token_type_ids.view(1, -1).repeat(beam_size, 1)
                else:
                    scores = self.forward(new_input_ids, new_token_type_ids)
                
                logit_score = torch.log_softmax(scores[:, -1], dim=-1)
                
                logit_score = output_scores.view(-1, 1) + logit_score # 累计得分
                ## 取topk的时候我们是展平了然后再去调用topk函数
                # 展平
                logit_score = logit_score.view(-1)
                hype_score, hype_pos = torch.topk(logit_score, beam_size)
                indice1 = (hype_pos // scores.shape[-1]) # 行索引
                indice2 = (hype_pos % scores.shape[-1]).long().reshape(-1, 1) # 列索引
               
                # 更新得分
                output_scores = hype_score
                output_ids = torch.cat([output_ids[indice1], indice2], dim=1).long()
                new_input_ids = torch.cat([token_ids, output_ids], dim=1)
                new_token_type_ids = torch.cat([token_type_ids, torch.ones_like(output_ids)], dim=1)

                end_counts = (output_ids == sep_id).sum(1)  # 统计出现的end标记
                best_one = output_scores.argmax()
                if end_counts[best_one] == 1:
                    # 说明出现终止了～
                    return output_ids[best_one][:-1]
                else :
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

    def beam_search_poem(self, text, token_ids, token_type_ids, word2ix, beam_size=1, device="cpu"):
        """
        beam-search操作
        """
        yayun_pos = []
        title = text.split("##")[0]
        if "五言律诗" in text:
            yayun_pos = [10, 22, 34, 46]
        elif "五言绝句" in text:
            yayun_pos = [10, 22]
        elif "七言律诗" in text:
            yayun_pos = [14, 30, 46, 62]
        elif "七言绝句" in text:
            yayun_pos = [14, 30]
        sep_id = word2ix["[SEP]"]
        douhao_id = word2ix["，"]# 逗号
        ix2word = {v: k for k, v in word2ix.items()}
        juhao_id = word2ix["。"]# 句号
        repeat_word = [[] for i in range(beam_size)]
        # 用来保存输出序列
        output_ids = torch.empty(1, 0, device=device, dtype=torch.long)
        last_chars = torch.empty(1, 0, device=device, dtype=torch.long)
        yayun_chars = (-1) * torch.ones(beam_size, dtype=torch.long)
        start = 0
        with torch.no_grad(): 
            output_scores = torch.zeros(token_ids.shape[0], device=device)
            for step in range(self.out_max_length):
                if step == 0:
                    scores = self.forward(token_ids, token_type_ids)
                    # 重复beam-size次 输入ids
                    token_ids = token_ids.view(1, -1).repeat(beam_size, 1)
                    token_type_ids = token_type_ids.view(1, -1).repeat(beam_size, 1)
                else:
                    scores = self.forward(new_input_ids, new_token_type_ids)
                
                logit_score = torch.log_softmax(scores[:, -1], dim=-1)
                
                for i, char in enumerate(last_chars):
                    
                    for word in repeat_word[i]:
                        logit_score[i, word] -= 5
                    for word in title:
                        ix = word2ix.get(word, -1)
                        if ix != -1:
                            logit_score[i, ix] += 2

                if step in yayun_pos:
                    # print("step is " + str(step))
                    # print("yayun_chars is " + str(yayun_chars))
                    for i, char in enumerate(last_chars):
                        if yayun_chars[i].item() != -1:
                            yayuns = yayun_list[yayun_chars[i].item()]
                            for char in yayuns:
                                ix = word2ix.get(char, -1)
                                if ix != -1:
                                    # print("char is " + str(char))
                                    logit_score[i, ix] += 10


                logit_score = output_scores.view(-1, 1) + logit_score # 累计得分
                ## 取topk的时候我们是展平了然后再去调用topk函数
                # 展平
                logit_score = logit_score.view(-1)
                hype_score, hype_pos = torch.topk(logit_score, beam_size)
                indice1 = (hype_pos // scores.shape[-1]) # 行索引
                indice2 = (hype_pos % scores.shape[-1]).long().reshape(-1, 1) # 列索引
                
                for index, each_out in zip(indice1, indice2):
                    index = index.item()
                    each_out = each_out.item()
                    
                    if each_out in repeat_word[index]:
                        pass 
                        # repeat_word[index].append(each_out)
                        # hype_score[index] -= 2 * repeat_word[index].count(each_out)
                    else :
                        repeat_word[index].append(each_out)
                    
                    if start < beam_size and each_out == douhao_id and len(last_chars) != 0:
                        start += 1
                        word = ix2word[last_chars[index].item()]# 找到上一个字符 记住其押韵情况
                        for i, each_yayun in enumerate(yayun_list):
                            if word in each_yayun:
                                yayun_chars[index] = i
                                break

                # 更新得分
                output_scores = hype_score

                last_chars = indice2

                output_ids = torch.cat([output_ids[indice1], indice2], dim=1).long()
                new_input_ids = torch.cat([token_ids, output_ids], dim=1)
                new_token_type_ids = torch.cat([token_type_ids, torch.ones_like(output_ids)], dim=1)

                end_counts = (output_ids == sep_id).sum(1)  # 统计出现的end标记
                best_one = output_scores.argmax()
                if end_counts[best_one] == 1:
                    # 说明出现终止了～
                    # print(repeat_word)
                    # print(yayun_chars)
                    return output_ids[best_one][:-1]
                else :
                    # 保留未完成部分
                    flag = (end_counts < 1)  # 标记未完成序列
                    if not flag.all():  # 如果有已完成的
                        token_ids = token_ids[flag]
                        token_type_ids = token_type_ids[flag]
                        last_chars = last_chars[flag]
                        yayun_chars = yayun_chars[flag]
                        new_input_ids = new_input_ids[flag]
                        new_token_type_ids = new_token_type_ids[flag]
                        output_ids = output_ids[flag]  # 扔掉已完成序列
                        output_scores = output_scores[flag]  # 扔掉已完成序列
                        end_counts = end_counts[flag]  # 扔掉已完成end计数
                        beam_size = flag.sum()  # topk相应变化
                        flag = flag.long()

                        new_repeat_word = []
                        for index, i in enumerate(flag):
                            if i.item() == 1:
                                new_repeat_word.append(repeat_word[index])
                     
                        repeat_word = new_repeat_word

            return output_ids[output_scores.argmax()]
    
    def beam_search_poem_v2(self, text, token_ids, token_type_ids, word2ix, beam_size=1, device="cpu"):
        """
        beam-search操作
        """
        yayun_pos = []
        if "五言律诗" in text:
            yayun_pos = [10, 22, 34, 46]
        elif "五言绝句" in text:
            yayun_pos = [10, 22]
        elif "七言律诗" in text:
            yayun_pos = [14, 30, 46, 62]
        elif "七言绝句" in text:
            yayun_pos = [14, 30]
        sep_id = word2ix["[SEP]"]
        douhao_id = word2ix["，"]# 逗号
        ix2word = {v: k for k, v in word2ix.items()}
        juhao_id = word2ix["。"]# 句号
        repeat_word = []
        # 用来保存输出序列
        output_ids = torch.empty(1, 0, device=device, dtype=torch.long)
        last_chars = torch.empty(1, 0, device=device, dtype=torch.long)
        yayun_chars = (-1) * torch.ones(beam_size, dtype=torch.long)
        start = 0
        with torch.no_grad(): 
            output_scores = torch.zeros(token_ids.shape[0], device=device)
            for step in range(self.out_max_length):
                if step == 0:
                    scores = self.forward(token_ids, token_type_ids)
                    # 重复beam-size次 输入ids
                    token_ids = token_ids.view(1, -1).repeat(beam_size, 1)
                    token_type_ids = token_type_ids.view(1, -1).repeat(beam_size, 1)
                else:
                    scores = self.forward(new_input_ids, new_token_type_ids)
                
                logit_score = torch.log_softmax(scores[:, -1], dim=-1)
                # if len(last_chars) != 0:
                #     logit_score[last_chars] -= 5
                for i, char in enumerate(last_chars):
                    logit_score[i, char] -= 2
                    for word in repeat_word:
                        logit_score[i, word] -= 1
                if step in yayun_pos:
                    # print("step is " + str(step))
                    # print("yayun_chars is " + str(yayun_chars))
                    for i, char in enumerate(last_chars):
                        if yayun_chars[i].item() != -1:
                            yayuns = yayun_list[yayun_chars[i].item()]
                            for char in yayuns:
                                ix = word2ix.get(char, -1)
                                if ix != -1:
                                    # print("char is " + str(char))
                                    logit_score[i, ix] += 3
                logit_score = output_scores.view(-1, 1) + logit_score # 累计得分
                ## 取topk的时候我们是展平了然后再去调用topk函数
                # 展平
                logit_score = logit_score.view(-1)
                hype_score, hype_pos = torch.topk(logit_score, beam_size)
                indice1 = (hype_pos // scores.shape[-1]) # 行索引
                indice2 = (hype_pos % scores.shape[-1]).long().reshape(-1, 1) # 列索引
                
                for index, each_out in zip(indice1, indice2):
                    index = index.item()
                    each_out = each_out.item()
                    
                    if each_out in repeat_word:
                        pass 
                        # repeat_word[index].append(each_out)
                        # hype_score[index] -= 2 * repeat_word[index].count(each_out)
                    else :
                        repeat_word.append(each_out)
                    
                    if start < beam_size and each_out == douhao_id and len(last_chars) != 0:
                        start += 1
                        word = ix2word[last_chars[index].item()]# 找到上一个字符 记住其押韵情况
                        for i, each_yayun in enumerate(yayun_list):
                            if word in each_yayun:
                                yayun_chars[index] = i
                                break

                # 更新得分
                output_scores = hype_score

                last_chars = indice2

                output_ids = torch.cat([output_ids[indice1], indice2], dim=1).long()
                new_input_ids = torch.cat([token_ids, output_ids], dim=1)
                new_token_type_ids = torch.cat([token_type_ids, torch.ones_like(output_ids)], dim=1)

                end_counts = (output_ids == sep_id).sum(1)  # 统计出现的end标记
                best_one = output_scores.argmax()
                if end_counts[best_one] == 1:
                    # 说明出现终止了～
                    # print(repeat_word)
                    # print(yayun_chars)
                    return output_ids[best_one]
                else :
                    # 保留未完成部分
                    flag = (end_counts < 1)  # 标记未完成序列
                    if not flag.all():  # 如果有已完成的
                        token_ids = token_ids[flag]
                        token_type_ids = token_type_ids[flag]
                        last_chars = last_chars[flag]
                        yayun_chars = yayun_chars[flag]
                        new_input_ids = new_input_ids[flag]
                        new_token_type_ids = new_token_type_ids[flag]
                        output_ids = output_ids[flag]  # 扔掉已完成序列
                        output_scores = output_scores[flag]  # 扔掉已完成序列
                        end_counts = end_counts[flag]  # 扔掉已完成end计数
                        beam_size = flag.sum()  # topk相应变化
                        flag = flag.long()


            return output_ids[output_scores.argmax()]


