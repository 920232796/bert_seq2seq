## bert 关系抽取模型
import torch 
import torch.nn as nn 
from bert_seq2seq.tokenizer import load_chinese_base_vocab, Tokenizer

class BertRelationExtrac(nn.Module):
    """
    """
    def __init__(self, word2ix, predicate_num, model_name="roberta"):
        super(BertRelationExtrac, self).__init__()
        
        self.predicate_num = predicate_num 
        config = ""
        if model_name == "roberta":
            from bert_seq2seq.model.roberta_model import BertModel, BertConfig, BertPredictionHeadTransform, BertLayerNorm
            config = BertConfig(len(word2ix))
            self.bert = BertModel(config)
            self.layer_norm = BertLayerNorm(config.hidden_size)
            self.layer_norm_cond = BertLayerNorm(config.hidden_size, conditional=True)
        elif model_name == "bert":
            from bert_seq2seq.model.bert_model import BertConfig, BertModel, BertPredictionHeadTransform, BertLayerNorm
            config = BertConfig(len(word2ix))
            self.bert = BertModel(config)
            self.layer_norm = BertLayerNorm(config.hidden_size)
            self.layer_norm_cond = BertLayerNorm(config.hidden_size, conditional=True)
        else :
            raise Exception("model_name_err")
        
        self.subject_pred = nn.Linear(config.hidden_size, 2)
        self.activation = nn.Sigmoid()
        self.object_pred = nn.Linear(config.hidden_size, 2 * self.predicate_num)
    
    def binary_crossentropy(self, labels, pred):
        labels = labels.float()
        loss = (-labels) * torch.log(pred) - (1.0 - labels) * torch.log(1.0 - pred)
        return loss

    def compute_total_loss(self, subject_pred, object_pred, subject_labels, object_labels):
        """
        计算loss
        """
        subject_loss = self.binary_crossentropy(subject_labels, subject_pred)
        subject_loss = torch.mean(subject_loss, dim=2)
        subject_loss = (subject_loss * self.target_mask).sum() / self.target_mask.sum()

        object_loss = self.binary_crossentropy(object_labels, object_pred)
        object_loss = torch.mean(object_loss, dim=3).sum(dim=2)
        object_loss = (object_loss * self.target_mask).sum() / self.target_mask.sum()

        return subject_loss + object_loss
    
    def extrac_subject(self, output, subject_ids):
        ## 抽取subject的向量表征
        batch_size = output.shape[0]
        hidden_size = output.shape[-1]
        start = torch.gather(output, index=subject_ids[:, :1].unsqueeze(1).expand((batch_size, 1, hidden_size)), dim=1)
        end = torch.gather(output, index=subject_ids[:, 1: ].unsqueeze(1).expand((batch_size, 1, hidden_size)), dim=1)
        subject = torch.cat((start, end), dim=-1)
        return subject[:, 0]

    def forward(self, text, subject_ids, position_enc=None, subject_labels=None, object_labels=None, use_layer_num=-1, device="cpu"):
        if use_layer_num != -1:
            if use_layer_num < 0 or use_layer_num > 7:
                # 越界
                raise Exception("层数选择错误，因为bert base模型共8层，所以参数只只允许0 - 7， 默认为-1，取最后一层")
        # 计算target mask
        text = text.to(device)
        subject_ids = subject_ids.to(device)

        self.target_mask = (text > 0).float()
        enc_layers, _ = self.bert(text, 
                                    output_all_encoded_layers=True)
        squence_out = enc_layers[use_layer_num] 

        transform_out = self.layer_norm(squence_out)
        subject_pred_out = self.subject_pred(transform_out)

        subject_pred_act = self.activation(subject_pred_out)
        subject_vec = self.extrac_subject(squence_out, subject_ids)
        object_layer_norm = self.layer_norm_cond([squence_out, subject_vec])
        object_pred_out = self.object_pred(object_layer_norm)
        object_pred_act = self.activation(object_pred_out)
        batch_size, seq_len, target_size = object_pred_act.shape

        object_pred_act = object_pred_act.reshape((batch_size, seq_len, int(target_size/2), 2))
        # print(object_pred_act.shape)
        predictions = object_pred_act
        if subject_labels is not None and object_labels is not None:
            ## 计算loss
            subject_labels = subject_labels.to(device)
            object_labels = object_labels.to(device)
            loss = self.compute_total_loss(subject_pred_act, object_pred_act, subject_labels, object_labels)
            return predictions, loss 
        else :
            return predictions

    def predict_subject(self, text, position_enc=None, use_layer_num=-1, device="cpu"):
        if use_layer_num != -1:
            if use_layer_num < 0 or use_layer_num > 7:
                # 越界
                raise Exception("层数选择错误，因为bert base模型共8层，所以参数只只允许0 - 7， 默认为-1，取最后一层")
        # 计算target mask
        text = text.to(device)

        self.target_mask = (text > 0).float()
        enc_layers, _ = self.bert(text, output_all_encoded_layers=True)
        squence_out = enc_layers[use_layer_num]

        transform_out = self.layer_norm(squence_out)
        subject_pred_out = self.subject_pred(transform_out)

        subject_pred_act = self.activation(subject_pred_out)

        return subject_pred_act

    def predict_object_predicate(self, text, subject_ids, position_enc=None, subject_labels=None, object_labels=None, use_layer_num=-1, device="cpu"):
        if use_layer_num != -1:
            if use_layer_num < 0 or use_layer_num > 7:
                # 越界
                raise Exception("层数选择错误，因为bert base模型共8层，所以参数只只允许0 - 7， 默认为-1，取最后一层")
        # 计算target mask
        text = text.to(device)
        subject_ids = subject_ids.to(device)

        self.target_mask = (text > 0).float()
        enc_layers, _ = self.bert(text, output_all_encoded_layers=True)
        squence_out = enc_layers[use_layer_num]

        subject_vec = self.extrac_subject(squence_out, subject_ids)
        object_layer_norm = self.layer_norm_cond([squence_out, subject_vec])
        object_pred_out = self.object_pred(object_layer_norm)
        object_pred_act = self.activation(object_pred_out)
        batch_size, seq_len, target_size = object_pred_act.shape

        object_pred_act = object_pred_act.reshape((batch_size, seq_len, int(target_size/2), 2))
        # print(object_pred_act.shape)
        predictions = object_pred_act
        return predictions
