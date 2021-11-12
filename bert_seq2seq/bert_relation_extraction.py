## bert 关系抽取模型
import torch 
import torch.nn as nn 
from bert_seq2seq.tokenizer import load_chinese_base_vocab, Tokenizer
from bert_seq2seq.basic_bert import BasicBert

class BertRelationExtrac(BasicBert):
    """
    """
    def __init__(self, word2ix, predicate_num, model_name="roberta"):
        super(BertRelationExtrac, self).__init__(word2ix=word2ix, model_name=model_name)
        
        self.predicate_num = predicate_num
        self.subject_pred_norm = nn.LayerNorm(self.config.hidden_size)
        self.subject_pred = nn.Linear(self.config.hidden_size, 2)
        self.activation = nn.Sigmoid()
        self.object_pred = nn.Linear(self.config.hidden_size, 2 * self.predicate_num)
        
    
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
        start_end = torch.gather(output, index=subject_ids.unsqueeze(-1).expand((batch_size, 2, hidden_size)), dim=1)
        subject = torch.cat((start_end[:, 0], start_end[:, 1]), dim=-1)
        return subject

    def forward(self, text, subject_ids, position_enc=None, subject_labels=None, object_labels=None, use_layer_num=-1):
        if use_layer_num != -1:
            
            raise Exception("目前 use_layer_num 只支持-1")
        # 计算target mask
        text = text.to(self.device)
        subject_ids = subject_ids.to(self.device)
        self.target_mask = (text > 0).float()
        enc_layers, _ = self.bert(text, 
                                    output_all_encoded_layers=True)

        squence_out = enc_layers[use_layer_num] 

        tokens_hidden_state, _ = self.cls(squence_out)

        subject_pred_out = self.subject_pred(self.subject_pred_norm(tokens_hidden_state))

        subject_pred_act = self.activation(subject_pred_out)

        subject_pred_act = subject_pred_act**2 

        subject_vec = self.extrac_subject(tokens_hidden_state, subject_ids)
        object_layer_norm = self.layer_norm_cond([tokens_hidden_state, subject_vec])
        object_pred_out = self.object_pred(object_layer_norm)
        object_pred_act = self.activation(object_pred_out)

        object_pred_act = object_pred_act**4

        batch_size, seq_len, target_size = object_pred_act.shape

        object_pred_act = object_pred_act.reshape((batch_size, seq_len, int(target_size/2), 2))
        predictions = object_pred_act
        if subject_labels is not None and object_labels is not None:
            ## 计算loss
            subject_labels = subject_labels.to(self.device)
            object_labels = object_labels.to(self.device)
            loss = self.compute_total_loss(subject_pred_act, object_pred_act, subject_labels, object_labels)
            return predictions, loss 
        else :
            return predictions

    def predict_subject(self, text,use_layer_num=-1):
        if use_layer_num != -1:
            
            raise Exception("use_layer_num目前只支持-1")
        text = text.to(self.device)

        self.target_mask = (text > 0).float()
        enc_layers, _ = self.bert(text, output_all_encoded_layers=True)
        squence_out = enc_layers[use_layer_num]
        tokens_hidden_state, _ = self.cls(squence_out)
        subject_pred_out = self.subject_pred(self.subject_pred_norm(tokens_hidden_state))
        subject_pred_act = self.activation(subject_pred_out)

        subject_pred_act = subject_pred_act**2

        # subject_pred_act = (subject_pred_act > 0.5).long() 
        return subject_pred_act

    def predict_object_predicate(self, text, subject_ids, use_layer_num=-1):
        if use_layer_num != -1:
            
                raise Exception("use_layer_num目前只支持-1")
        # 计算target mask
        text = text.to(self.device)
        subject_ids = subject_ids.to(self.device)

        enc_layers, _ = self.bert(text, output_all_encoded_layers=True)
        squence_out = enc_layers[use_layer_num]
        tokens_hidden_state, _ = self.cls(squence_out)

        subject_vec = self.extrac_subject(tokens_hidden_state, subject_ids)
        object_layer_norm = self.layer_norm_cond([tokens_hidden_state, subject_vec])
        object_pred_out = self.object_pred(object_layer_norm)
        object_pred_act = self.activation(object_pred_out)

        object_pred_act = object_pred_act**4 

        batch_size, seq_len, target_size = object_pred_act.shape
        object_pred_act = object_pred_act.view((batch_size, seq_len, int(target_size/2), 2))
        predictions = object_pred_act
        return predictions