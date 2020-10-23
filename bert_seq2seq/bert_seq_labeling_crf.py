## bert encoder模型
import torch 
import torch.nn as nn 
from bert_seq2seq.tokenizer import load_chinese_base_vocab, Tokenizer
from bert_seq2seq.model.crf import CRFLayer

class BertSeqLabelingCRF(nn.Module):
    """
    """
    def __init__(self, word2ix, target_size, model_name="roberta"):
        super(BertSeqLabelingCRF, self).__init__()
        self.target_size = target_size
        config = ""
        if model_name == "roberta":
            from bert_seq2seq.model.roberta_model import BertModel, BertConfig, BertPredictionHeadTransform
            config = BertConfig(len(word2ix))
            self.bert = BertModel(config)
            self.transform = BertPredictionHeadTransform(config)
        elif model_name == "bert":
            from bert_seq2seq.model.bert_model import BertConfig, BertModel, BertPredictionHeadTransform
            config = BertConfig(len(word2ix))
            self.bert = BertModel(config)
            self.transform = BertPredictionHeadTransform(config)
        else :
            raise Exception("model_name_err")
        
        self.final_dense = nn.Linear(config.hidden_size, self.target_size)
        self.crf_layer = CRFLayer(self.target_size)

        # self.activation = nn.Sigmoid()
    
    def compute_loss(self, predictions, labels):
        """
        计算loss
        """
        loss = self.crf_layer(predictions, labels, self.target_mask)
        
        return loss.mean()
    
    def forward(self, text, position_enc=None, labels=None, use_layer_num=-1):
        if use_layer_num != -1:
            if use_layer_num < 0 or use_layer_num > 7:
                # 越界
                raise Exception("层数选择错误，因为bert base模型共8层，所以参数只只允许0 - 7， 默认为-1，取最后一层")
        # 计算target mask
        self.target_mask = (text > 0).float()
        enc_layers, _ = self.bert(text, 
                                    output_all_encoded_layers=True)
        squence_out = enc_layers[use_layer_num] 

        transform_out = self.transform(squence_out)
        # print(cls_token)
        predictions = self.final_dense(transform_out)

        if labels is not None:
            ## 计算loss
            loss = self.compute_loss(predictions, labels)
            return predictions, loss 
        else :
            return predictions
