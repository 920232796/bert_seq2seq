
import torch
import torch.nn as nn 

class BasicBert(nn.Module):
    def __init__(self, word2ix, model_name="roberta"):
        super().__init__()
        self.config = ""
        self.word2ix = word2ix
        self.model_name = model_name
        if model_name == "roberta":
            from bert_seq2seq.model.roberta_model import BertModel, BertConfig, BertLayerNorm, BertPredictionHeadTransform,BertLMPredictionHead
        elif model_name == "bert":
            from bert_seq2seq.model.bert_model import BertConfig, BertModel, BertLayerNorm, BertPredictionHeadTransform,BertLMPredictionHead
        elif model_name == "nezha":
            from bert_seq2seq.model.nezha_model import BertConfig, BertModel, BertLayerNorm, BertPredictionHeadTransform,BertLMPredictionHead
        else:
            raise Exception("model_name_err")

        self.config = BertConfig(vocab_size=len(self.word2ix))
        self.bert = BertModel(self.config)
        self.layer_norm = BertLayerNorm(self.config.hidden_size)
        self.layer_norm_cond = BertLayerNorm(self.config.hidden_size, conditional=True)
        self.transform = BertPredictionHeadTransform(self.config)
        self.decoder = BertLMPredictionHead(self.config, self.bert.embeddings.word_embeddings.weight)

        self.device = torch.device("cpu")

    def load_pretrain_params(self, pretrain_model_path, keep_tokens=None):
        checkpoint = torch.load(pretrain_model_path, map_location=self.device)

        checkpoint = {k: v for k, v in checkpoint.items()
                                            if k[:4] == "bert" and "pooler" not in k}
        if keep_tokens is not None:
            ## 说明精简词表了，embeedding层也要过滤下
            embedding_weight_name = "bert.embeddings.word_embeddings.weight"
            checkpoint[embedding_weight_name] = checkpoint[embedding_weight_name][keep_tokens]
            
        self.load_state_dict(checkpoint, strict=False)
        torch.cuda.empty_cache()
        print("{} loaded!".format(pretrain_model_path))

    def load_all_params(self, model_path, device="cuda"):

        checkpoint = torch.load(model_path, map_location=device)
        self.load_state_dict(checkpoint, strict=False)
        torch.cuda.empty_cache()
        print(str(model_path) + " loaded!")

    def forward(self, x):
        raise NotImplemented

    def set_device(self, device):
        self.device = torch.device(device)
        self.to(device)
        
    def save_all_params(self, save_path):
        if self.model_name == "nezha":
            # 不要保存相对位置编码权重
            checkpoints = {k: v for k, v in self.state_dict().items()
                                        if "relative" not in k}
            torch.save(checkpoints, save_path)
            return
        torch.save(self.state_dict(), save_path)

class BasicGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device("cpu")

    def load_pretrain_params(self, pretrain_model_path):
        checkpoint = torch.load(pretrain_model_path, map_location=self.device)
        checkpoint = {"model." + k: v for k, v in checkpoint.items()}

        self.load_state_dict(checkpoint, strict=True)
        torch.cuda.empty_cache()
        print("{} loaded!".format(pretrain_model_path))

    def load_all_params(self, model_path, device="cuda"):
        checkpoint = torch.load(model_path, map_location=device)
        self.load_state_dict(checkpoint, strict=False)
        torch.cuda.empty_cache()
        print(str(model_path) + " loaded!")

    def forward(self, x):
        raise NotImplemented

    def set_device(self, device):
        self.device = torch.device(device)
        self.to(device)
        
    def save_all_params(self, save_path):
        torch.save(self.state_dict(), save_path)


class BasicT5(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device("cpu")

    def load_pretrain_params(self, pretrain_model_path):
        checkpoint = torch.load(pretrain_model_path, map_location=self.device)
        checkpoint = {"model." + k: v for k, v in checkpoint.items()}

        self.load_state_dict(checkpoint, strict=True)
        torch.cuda.empty_cache()
        print("{} loaded!".format(pretrain_model_path))

    def load_all_params(self, model_path, device="cuda"):
        checkpoint = torch.load(model_path, map_location=device)
        self.load_state_dict(checkpoint, strict=False)
        torch.cuda.empty_cache()
        print(str(model_path) + " loaded!")

    def forward(self, x):
        raise NotImplemented

    def set_device(self, device):
        self.device = torch.device(device)
        self.to(device)

    def save_all_params(self, save_path):
        torch.save(self.state_dict(), save_path)

class BasicBart(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device("cpu")

    def load_pretrain_params(self, pretrain_model_path):
        checkpoint = torch.load(pretrain_model_path, map_location=self.device)
        checkpoint = {"model." + k: v for k, v in checkpoint.items()}

        self.load_state_dict(checkpoint, strict=False)
        torch.cuda.empty_cache()
        print("{} loaded!".format(pretrain_model_path))

    def load_all_params(self, model_path, device="cuda"):
        checkpoint = torch.load(model_path, map_location=device)
        self.load_state_dict(checkpoint, strict=False)
        torch.cuda.empty_cache()
        print(str(model_path) + " loaded!")

    def forward(self, x):
        raise NotImplemented

    def set_device(self, device):
        self.device = torch.device(device)
        self.to(device)

    def save_all_params(self, save_path):
        torch.save(self.state_dict(), save_path)

