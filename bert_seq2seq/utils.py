import torch
from bert_seq2seq.seq2seq_model import Seq2SeqModel
from bert_seq2seq.bert_cls_classifier import BertClsClassifier
from bert_seq2seq.bert_seq_labeling import BertSeqLabeling
from bert_seq2seq.bert_seq_labeling_crf import BertSeqLabelingCRF
from bert_seq2seq.bert_relation_extraction import BertRelationExtrac

def load_bert(word2ix, model_name="roberta", model_class="seq2seq", target_size=0):
    """
    model_path: 模型位置
    这是个统一的接口，用来加载模型的
    model_class : seq2seq or encoder
    """
    if model_class == "seq2seq":
        bert_model = Seq2SeqModel(word2ix, model_name=model_name)
        return bert_model
    elif model_class == "cls":
        if target_size == 0:
            raise Exception("必须传入参数 target_size，才能确定预测多少分类")
        bert_model = BertClsClassifier(word2ix, target_size, model_name=model_name)
        return bert_model
    elif model_class == "sequence_labeling":
        ## 序列标注模型
        if target_size == 0:
            raise Exception("必须传入参数 target_size，才能确定预测多少分类")
        bert_model = BertSeqLabeling(word2ix, target_size, model_name=model_name)
        return bert_model
    elif model_class == "sequence_labeling_crf":
        # 带有crf层的序列标注模型
        if target_size == 0:
            raise Exception("必须传入参数 target_size，才能确定预测多少分类")
        bert_model = BertSeqLabelingCRF(word2ix, target_size, model_name=model_name)
        return bert_model
    elif model_class == "relation_extrac":
        if target_size == 0:
            raise Exception("必须传入参数 target_size 表示预测predicate的种类")
        bert_model = BertRelationExtrac(word2ix, target_size, model_name=model_name)
        return bert_model
    else :
        raise Exception("model_name_err")

def load_model_params(model, pretrain_model_path, keep_tokens=None):
        
        checkpoint = torch.load(pretrain_model_path)
        # 模型刚开始训练的时候, 需要载入预训练的BERT

        # checkpoint = {k[5:]: v for k, v in checkpoint.items()
        checkpoint = {k: v for k, v in checkpoint.items()
                                            if k[:4] == "bert" and "pooler" not in k}
        if keep_tokens is not None:
            ## 说明精简词表了，embeedding层也要过滤下
            embedding_weight_name = "bert.embeddings.word_embeddings.weight"
            
            checkpoint[embedding_weight_name] = checkpoint[embedding_weight_name][keep_tokens]
            
        model.load_state_dict(checkpoint, strict=False)
        torch.cuda.empty_cache()
        print("{} loaded!".format(pretrain_model_path))

def load_recent_model(model, recent_model_path):
    checkpoint = torch.load(recent_model_path)
    model.load_state_dict(checkpoint)
    torch.cuda.empty_cache()
    print(str(recent_model_path) + " loaded!")


