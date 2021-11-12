## bert encoder模型
import torch 
import torch.nn as nn 
from bert_seq2seq.tokenizer import Tokenizer
from bert_seq2seq.model.crf import CRFLayer
from bert_seq2seq.basic_bert import BasicBert

class BertSeqLabelingCRF(BasicBert):
    """
    """
    def __init__(self, word2ix, target_size, model_name="roberta"):
        super(BertSeqLabelingCRF, self).__init__(word2ix=word2ix, model_name=model_name)
        self.target_size = target_size

        self.final_dense = nn.Linear(self.config.hidden_size, self.target_size)
        self.crf_layer = CRFLayer(self.target_size)
    
    def compute_loss(self, predictions, labels):
        """
        计算loss
        """
        loss = self.crf_layer(predictions, labels, self.target_mask)
        
        return loss.mean()
    
    def forward(self, text, position_enc=None, labels=None, use_layer_num=-1):
        if use_layer_num != -1:
            # 越界
            raise Exception("use_layer_num目前只支持-1")
        # 计算target mask
        self.target_mask = (text > 0).float().to(self.device)
        text = text.to(self.device)
        if position_enc is not None :
            position_enc = position_enc.to(self.device)
        if labels is not None :
            labels = labels.to(self.device)
        enc_layers, _ = self.bert(text, 
                                    output_all_encoded_layers=True)
        squence_out = enc_layers[use_layer_num] 

        tokens_hidden_state, _ = self.cls(squence_out)
        # print(cls_token)
        predictions = self.final_dense(tokens_hidden_state)

        if labels is not None:
            ## 计算loss
            loss = self.compute_loss(predictions, labels)
            return predictions, loss 
        else :
            return predictions
