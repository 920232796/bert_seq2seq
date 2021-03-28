
from bert_seq2seq.tokenizer import load_chinese_base_vocab, T5PegasusTokenizer

from bert_seq2seq.model.t5_model import T5ForConditionalGeneration, T5Config
import torch
import torch.nn.functional as F

def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits

def sample_generate_encoder_decoder(model, text, tokenizer, eos_id, bos_id, device, input_max_length=256, out_max_length=200, top_k=30, top_p=0.0,
                                    add_eos=False):
    token_out = tokenizer.encode(text, max_length=input_max_length)
    if len(token_out) == 2:
        token_ids = token_out[0]
    else:
        token_ids = token_out
    if add_eos:
        token_ids = token_ids + [eos_id]
    token_ids = torch.tensor(token_ids, device=device, dtype=torch.long).view(1, -1)
    output_ids = []

    input_decoder_ids = torch.tensor(bos_id, device=device, dtype=torch.long).view(1, -1)
    with torch.no_grad():
        for step in range(out_max_length):
            scores = model(input_ids=token_ids, decoder_input_ids=input_decoder_ids)[0]
            logit_score = torch.log_softmax(scores[:, -1], dim=-1).squeeze(0)
            # if self.tokenizer.unk_token_id is not None:
            #     logit_score[self.tokenizer.unk_token_id] = -float('Inf')
            filtered_logits = top_k_top_p_filtering(logit_score, top_k=top_k, top_p=top_p)
            next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
            if eos_id == next_token.item():
                break
                # pass
            output_ids.append(next_token.item())
            # token_ids = torch.cat((token_ids, next_token.long().unsqueeze(0)), dim=1)
            input_decoder_ids = torch.cat((input_decoder_ids, next_token.long().unsqueeze(0)), dim=1)

    return tokenizer.decode(output_ids)

if __name__ == '__main__':
    config = T5Config()
    model = T5ForConditionalGeneration(config)

    checpoints = torch.load("./state_dict/t5-chinese/pytorch_model.bin")

    model.load_state_dict(checpoints)

    word2ix = load_chinese_base_vocab("./state_dict/t5-chinese/vocab.txt")
    tokenizer = T5PegasusTokenizer(word2ix)

    text = '从那之后，一发不可收拾。此后的近百年间，一共有十七位新娘在与君山一带失踪。有时十几年相安无事，有时短短一个月内失踪两名。一个恐怖传说迅速传开：与君山里住着一位鬼新郎，若是他看中了一位女子，便会在她出嫁的路上将她掳走，再把送亲的队伍吃掉。'
    out = sample_generate_encoder_decoder(model, text, tokenizer, word2ix["[SEP]"], word2ix["[CLS]"], device="cpu")
    print(out)