
import torch
import torch.nn as nn 

class BasicBert(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device("cpu")

    def load_pretrain_params(self, pretrain_model_path, keep_tokens=None):
        checkpoint = torch.load(pretrain_model_path, map_location=self.device)
        # 模型刚开始训练的时候, 需要载入预训练的BERT

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
        self.load_state_dict(checkpoint)
        torch.cuda.empty_cache()
        print(str(model_path) + " loaded!")

    def forward(self, x):
        raise NotImplemented

    def set_device(self, device):
        self.device = torch.device(device)
        self.to(device)
        
    def save_all_params(self, save_path):
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
        self.load_state_dict(checkpoint)
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
        self.load_state_dict(checkpoint)
        torch.cuda.empty_cache()
        print(str(model_path) + " loaded!")

    def forward(self, x):
        raise NotImplemented

    def set_device(self, device):
        self.device = torch.device(device)
        self.to(device)

    def save_all_params(self, save_path):
        torch.save(self.state_dict(), save_path)

