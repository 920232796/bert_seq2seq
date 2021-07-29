## bert encoder模型
import torch
import torch.nn as nn
from bert_seq2seq.tokenizer import Tokenizer
from bert_seq2seq.basic_bert import BasicBert

class BertClsClassifier(BasicBert):
    """
    """
    def __init__(self, word2ix, target_size, model_name="roberta"):
        super(BertClsClassifier, self).__init__(word2ix=word2ix, model_name=model_name)
        self.target_size = target_size
        self.final_dense = nn.Linear(self.config.hidden_size, self.target_size)
    
    def compute_loss(self, predictions, labels):
        """
        计算loss
        predictions: (batch_size, 1)
        """
        predictions = predictions.view(-1, self.target_size)
        labels = labels.view(-1)
        loss = nn.CrossEntropyLoss(reduction="mean")
        return loss(predictions, labels)
    
    def forward(self, text, position_enc=None, labels=None, use_layer_num=-1):
        if use_layer_num != -1:
            if use_layer_num < 0 or use_layer_num > 7:
                # 越界
                raise Exception("层数选择错误，因为bert base模型共8层，所以参数只只允许0 - 7， 默认为-1，取最后一层")
        text = text.to(self.device)
        if position_enc is not None:
            position_enc = position_enc.to(self.device)
        if labels is not None:
            labels = labels.to(self.device)
        enc_layers, _ = self.bert(text, 
                                    output_all_encoded_layers=True)
        squence_out = enc_layers[use_layer_num] 

        cls_token = squence_out[:, 0]# 取出cls向量 进行分类
        # print(cls_token)
        predictions = self.final_dense(cls_token)
        if labels is not None:
            ## 计算loss
            loss = self.compute_loss(predictions, labels)
            return predictions, loss 
        else :
            return predictions
