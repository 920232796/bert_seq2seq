import torch 
from torch.utils.data import DataLoader, Dataset


def gpt_collate_fn(batch):
    """
    动态padding， batch为一部分sample
    """

    def padding(indice, max_length, pad_idx=0):
        """
        pad 函数
        """
        pad_indice = [item + [pad_idx] * max(0, max_length - len(item)) for item in indice]
        return torch.tensor(pad_indice)

    token_ids = [data["token_ids"] for data in batch]
    max_length = max([len(t) for t in token_ids])

    token_ids_padded = padding(token_ids, max_length)
    target_ids_padded = token_ids_padded.clone()
    target_ids_padded[target_ids_padded == 0] = -100

    return token_ids_padded, target_ids_padded

def bert_seq2seq_collate_fn(batch):
    """
    动态padding， batch为一部分sample
    """

    def padding(indice, max_length, pad_idx=0):
        """
        pad 函数
        """
        pad_indice = [item + [pad_idx] * max(0, max_length - len(item)) for item in indice]
        return torch.tensor(pad_indice)

    token_ids = [data["token_ids"] for data in batch]
    max_length = max([len(t) for t in token_ids])
    token_type_ids = [data["token_type_ids"] for data in batch]

    token_ids_padded = padding(token_ids, max_length)
    token_type_ids_padded = padding(token_type_ids, max_length)
    target_ids_padded = token_ids_padded[:, 1:].contiguous()

    return token_ids_padded, token_type_ids_padded, target_ids_padded

def bert_cls_collate_fn(batch):
    """
    动态padding， batch为一部分sample
    """

    def padding(indice, max_length, pad_idx=0):
        """
        pad 函数
        """
        pad_indice = [item + [pad_idx] * max(0, max_length - len(item)) for item in indice]
        return torch.tensor(pad_indice)

    token_ids = [data["token_ids"] for data in batch]
    max_length = max([len(t) for t in token_ids])
    token_type_ids = [data["token_type_ids"] for data in batch]
    target_ids = [data["target_id"] for data in batch]
    target_ids = torch.tensor(target_ids, dtype=torch.long)

    token_ids_padded = padding(token_ids, max_length)
    token_type_ids_padded = padding(token_type_ids, max_length)
    # target_ids_padded = token_ids_padded[:, 1:].contiguous()

    return token_ids_padded, token_type_ids_padded, target_ids

def bert_squence_label_collate_fn(batch):
    """
    动态padding， batch为一部分sample
    """

    def padding(indice, max_length, pad_idx=0):
        """
        pad 函数
        """
        pad_indice = [item + [pad_idx] * max(0, max_length - len(item)) for item in indice]
        return torch.tensor(pad_indice)

    token_ids = [data["token_ids"] for data in batch]
    
    max_length = max([len(t) for t in token_ids])
    token_type_ids = [data["token_type_ids"] for data in batch]
    target_ids = [data["target_id"] for data in batch]
  
    token_ids_padded = padding(token_ids, max_length)
    token_type_ids_padded = padding(token_type_ids, max_length)
    # target_ids_padded = token_ids_padded[:, 1:].contiguous()
    target_ids_padded = padding(target_ids, max_length)

    return token_ids_padded, token_type_ids_padded, target_ids_padded

class AbstractDataset(Dataset):

    def __init__(self, model_name, model_class, collate_fn=None) -> None:
        super().__init__()
        if model_name == "gpt2":
            self.collate_fn = gpt_collate_fn
        if model_name == "bert" or model_name == "roberta" or model_name == "roberta-large" or model_name == "nezha":
            if model_class == "seq2seq":
                self.collate_fn = bert_seq2seq_collate_fn
            elif model_class == "cls":
                self.collate_fn = bert_cls_collate_fn
            elif model_class == "sequence_labeling_crf" or model_class == "sequence_labeling":
                self.collate_fn = bert_squence_label_collate_fn
                

        if collate_fn is not None :
            self.collate_fn = collate_fn

    
    def __getitem__(self, index):
        return NotImplemented
    
    def __len__(self):
        return NotImplemented