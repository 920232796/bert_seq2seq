## bert encoder模型
import torch 
import torch.nn as nn 
from bert_seq2seq.tokenizer import load_chinese_base_vocab, Tokenizer
from bert_seq2seq.basic_bert import BasicBert

class BertSeqLabeling(BasicBert):
    """
    """
    def __init__(self, word2ix, target_size, model_name="roberta"):
        super(BertSeqLabeling, self).__init__(word2ix=word2ix, model_name=model_name)
        self.target_size = target_size

        
        self.final_dense = nn.Linear(self.config.hidden_size, self.target_size)
    
    def compute_loss(self, predictions, labels):
        """
        计算loss
        predictions: (batch_size, 1)
        """
        predictions = predictions.view(-1, self.target_size)
        labels = labels.view(-1)
        self.target_mask = self.target_mask.view(-1)
        loss = nn.CrossEntropyLoss(reduction="none")
        return (loss(predictions, labels) * self.target_mask).sum() / self.target_mask.sum()
    
    def forward(self, text, position_enc=None, labels=None, use_layer_num=-1):
        if use_layer_num != -1:
            if use_layer_num < 0 or use_layer_num > 7:
                # 越界
                raise Exception("层数选择错误，因为bert base模型共8层，所以参数只只允许0 - 7， 默认为-1，取最后一层")
        self.target_mask = (text > 0).float().to(self.device)
        text = text.to(self.device)
        if position_enc is not None:
            position_enc = position_enc.to(self.device)
        if labels is not None:
            labels = labels.to(self.device)
            
        enc_layers, _ = self.bert(text, 
                                    output_all_encoded_layers=True)
        squence_out = enc_layers[use_layer_num] 

        tokens_hidden_state, _ = self.cls(squence_out)
        predictions = self.final_dense(tokens_hidden_state)
        if labels is not None:
            ## 计算loss
            loss = self.compute_loss(predictions, labels)
            return predictions, loss 
        else :
            return predictions
