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

class SimBertModel(BasicBert):
    """
    """
    def __init__(self, word2ix, model_name="roberta", tokenizer=None):
        super(SimBertModel, self).__init__()
        self.word2ix = word2ix
        if tokenizer is None:
            self.tokenizer = Tokenizer(word2ix)
        else:
            self.tokenizer = tokenizer
        config = ""
        if model_name == "roberta":
            from bert_seq2seq.model.roberta_model import BertModel, BertConfig, BertLMPredictionHead
            config = BertConfig(len(word2ix))
            self.bert = BertModel(config)
            self.decoder = BertLMPredictionHead(config, self.bert.embeddings.word_embeddings.weight)
        elif model_name == "bert":
            from bert_seq2seq.model.bert_model import BertConfig, BertModel, BertLMPredictionHead
            config = BertConfig(len(word2ix))
            self.bert = BertModel(config)
            self.decoder = BertLMPredictionHead(config, self.bert.embeddings.word_embeddings.weight)
        else :
            raise Exception("model_name_err")
            
        self.hidden_dim = config.hidden_size
        self.vocab_size = len(word2ix)


    def compute_loss(self, predictions, labels, target_mask):
        loss1 = self.compute_loss_of_seq2seq(predictions, labels, target_mask)
        loss2 = self.compute_loss_of_similarity(predictions[:, 0]) ## 拿出cls向量
        return loss1 + loss2

    def compute_loss_of_seq2seq(self, predictions, labels, target_mask):
        predictions = predictions.view(-1, self.vocab_size)
        labels = labels.view(-1)
        target_mask = target_mask.view(-1).float()
        loss = nn.CrossEntropyLoss(ignore_index=0, reduction="none")
        return (loss(predictions, labels) * target_mask).sum() / target_mask.sum()  ## 通过mask 取消 pad 和句子a部分预测的影响

    def compute_loss_of_similarity(self, y_pred):

        y_true = self.get_labels_of_similarity(y_pred)  # 构建标签
        y_true = y_true.to(self.device)
        norm_a = torch.nn.functional.normalize(y_pred, dim=-1, p=2)
        # y_pred = K.l2_normalize(y_pred, axis=1)  # 句向量归一化
        similarities = norm_a.matmul(norm_a.t())

        # similarities = K.dot(y_pred, K.transpose(y_pred))  # 相似度矩阵
        similarities = similarities - (torch.eye(y_pred.shape[0]) * 1e12).to(self.device)  # 排除对角线
        similarities = similarities * 30  # scale
        similarities = similarities
        loss_f = nn.CrossEntropyLoss()
        loss = loss_f(similarities, y_true)
        # loss = K.categorical_crossentropy(
        #     y_true, similarities, from_logits=True
        # )
        return loss

    def get_labels_of_similarity(self, y_pred):
        idxs = torch.arange(0, y_pred.shape[0])
        idxs_1 = idxs[None, :]
        idxs_2 = (idxs + 1 - idxs % 2 * 2)[:, None]
        labels = (idxs_1 == idxs_2).float().argmax(dim=-1).long()
        return labels

    def forward(self, input_tensor, token_type_id, position_enc=None, labels=None):
        ## 传入输入，位置编码，token type id ，还有句子a 和句子b的长度，注意都是传入一个batch数据
        ##  传入的几个值，在seq2seq 的batch iter 函数里面都可以返回
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
        a_mask = ones.tril() # 下三角矩阵
        s_ex12 = token_type_id.unsqueeze(1).unsqueeze(2).float()
        s_ex13 = token_type_id.unsqueeze(1).unsqueeze(3).float()
        a_mask = (1.0 - s_ex12) * (1.0 - s_ex13) + s_ex13 * a_mask 
            
        enc_layers, _ = self.bert(input_tensor, position_ids=position_enc, token_type_ids=token_type_id, attention_mask=a_mask, 
                                    output_all_encoded_layers=True)
        squence_out = enc_layers[-1] ## 取出来最后一层输出

        predictions = self.decoder(squence_out)

        if labels is not None:
            ## 计算loss
            ## 需要构建特殊的输出mask 才能计算正确的loss
            # 预测的值不用取最后sep符号的结果 因此是到-1
            predictions = predictions[:, :-1].contiguous()
            target_mask = token_type_id[:, 1:].contiguous()
            loss = self.compute_loss(predictions, labels, target_mask)
            return predictions, loss 
        else :
            return predictions

    
    def generate(self, text, out_max_length=40, beam_size=1, is_poem=False, max_length=256):
        # 对 一个 句子生成相应的结果
        ## 通过输出最大长度得到输入的最大长度，这里问题不大，如果超过最大长度会进行截断
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


    def sample_generate(self, text, out_max_length=40, top_k=30, top_p=0.0, max_length=256):
        input_max_length = max_length - out_max_length
        token_ids, token_type_ids = self.tokenizer.encode(text, max_length=input_max_length)

        token_ids = torch.tensor(token_ids, device=self.device, dtype=torch.long).view(1, -1)
        token_type_ids = torch.tensor(token_type_ids, device=self.device, dtype=torch.long).view(1, -1)
        device = self.device
        output_ids = []
        sep_id = self.word2ix["[SEP]"]
        with torch.no_grad(): 
            for step in range(out_max_length):
                scores = self.forward(token_ids, token_type_ids)
                logit_score = torch.log_softmax(scores[:, -1], dim=-1).squeeze(0)
                logit_score[self.word2ix["[UNK]"]] = -float('Inf')
                filtered_logits = top_k_top_p_filtering(logit_score, top_k=top_k, top_p=top_p)
                next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
                if sep_id == next_token.item():
                    break
                output_ids.append(next_token.item())
                token_ids = torch.cat((token_ids, next_token.long().unsqueeze(0)), dim=1)
                token_type_ids = torch.cat([token_type_ids, torch.ones((1, 1), device=device, dtype=torch.long)], dim=1)

        return self.tokenizer.decode(np.array(output_ids))

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


