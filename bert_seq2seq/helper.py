from typing import List
import torch 


class LogitsProcessor:
    """Abstract base class for all logit processors that can be applied during generation."""

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """Torch method for processing logits."""
        raise NotImplementedError(
            f"{self.__class__} is an abstract class. Only classes inheriting this class can be called."
        )


class RepetitionPenaltyLogitsProcessor(LogitsProcessor):
    r"""
    :class:`transformers.LogitsProcessor` enforcing an exponential penalty on repeated sequences.
    Args:
        repetition_penalty (:obj:`float`):
            The parameter for repetition penalty. 1.0 means no penalty. See `this paper
            <https://arxiv.org/pdf/1909.05858.pdf>`__ for more details.
    """

    def __init__(self, penalty: float):
        if not isinstance(penalty, float) or not (penalty > 0):
            raise ValueError(f"`penalty` has to be a strictly positive float, but is {penalty}")

        self.penalty = penalty

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        
        score = torch.gather(scores, 1, input_ids)

        # if score < 0 then repetition penalty has to be multiplied to reduce the previous token probability
        score = torch.where(score < 0, score * self.penalty, score / self.penalty)

        scores.scatter_(1, input_ids, score)
        return scores

class TemperatureLogitsProcessor(LogitsProcessor):
    r"""
    :class:`transformers.LogitsWarper` for temperature (exponential scaling output probability distribution).
    Args:
        temperature (:obj:`float`):
            The value used to module the logits distribution.
    """

    def __init__(self, temperature: float):
        if not isinstance(temperature, float) or not (temperature > 0):
            raise ValueError(f"`temperature` has to be a strictly positive float, but is {temperature}")

        self.temperature = temperature

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.FloatTensor:
        scores = scores / self.temperature
        return scores


class TopPLogitsProcessor(LogitsProcessor):
    """
    :class:`transformers.LogitsWarper` that performs top-p, i.e. restricting to top tokens summing to prob_cut_off <=
    prob_cut_off.
    Args:
        top_p (:obj:`float`):
            If set to < 1, only the most probable tokens with probabilities that add up to top_p or higher are
            kept for generation.
        filter_value (:obj:`float`, `optional`, defaults to :obj:`-float("Inf")`):
            All filtered values will be set to this float value.
        min_tokens_to_keep (:obj:`int`, `optional`, defaults to 1):
            Minimum number of tokens that cannot be filtered.
    """

    def __init__(self, top_p: float, filter_value: float = -float("Inf"), min_tokens_to_keep: int = 1):
        top_p = float(top_p)
        if top_p < 0 or top_p > 1.0:
            raise ValueError(f"`top_p` has to be a float > 0 and < 1, but is {top_p}")

        self.top_p = top_p
        self.filter_value = filter_value
        self.min_tokens_to_keep = min_tokens_to_keep

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        sorted_logits, sorted_indices = torch.sort(scores, descending=True)
        # print(sorted_logits.softmax(dim=-1))
        cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)

        # Remove tokens with cumulative top_p above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > self.top_p
        if self.min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., : self.min_tokens_to_keep - 1] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        scores = scores.masked_fill(indices_to_remove, self.filter_value)
        return scores

class TopKLogitsProcessor(LogitsProcessor):
    r"""
    :class:`transformers.LogitsWarper` that performs top-k, i.e. restricting to the k highest probability elements.
    Args:
        top_k (:obj:`int`):
            The number of highest probability vocabulary tokens to keep for top-k-filtering.
        filter_value (:obj:`float`, `optional`, defaults to :obj:`-float("Inf")`):
            All filtered values will be set to this float value.
        min_tokens_to_keep (:obj:`int`, `optional`, defaults to 1):
            Minimum number of tokens that cannot be filtered.
    """

    def __init__(self, top_k: int, filter_value: float = -float("Inf"), min_tokens_to_keep: int = 1):
        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError(f"`top_k` has to be a strictly positive integer, but is {top_k}")

        self.top_k = top_k
        self.filter_value = filter_value
        self.min_tokens_to_keep = min_tokens_to_keep

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        top_k = min(max(self.top_k, self.min_tokens_to_keep), scores.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = scores < torch.topk(scores, top_k)[0][..., -1, None]
        scores = scores.masked_fill(indices_to_remove, self.filter_value)
        return scores


class ListProcessor(LogitsProcessor):
    def __init__(self, list_processor: List[LogitsProcessor]) -> None:
        super().__init__()
        self.list_processor = list_processor
    
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        
        for processor in self.list_processor:
            scores = processor(input_ids, scores)
        
        return scores
    


if __name__ == "__main__":
    print("hello world")
    input_ids = torch.tensor([[1, 2, 0, 1]])
    scores = torch.tensor([[-10, -5, -3, -1]], dtype=torch.float32)
    
    # temp = TemperatureLogitsProcessor(10.0)

    # top_p = TopPLogitsProcessor(top_p=0.5)

    # top_k = TopKLogitsProcessor(top_k=1)

    # scores = temp(input_ids, scores)
    # print(scores)

    # scores = top_p(input_ids, scores)

    # print(scores)

    # scores = top_k(input_ids, scores)

    # print(scores)

    list_processor = ListProcessor([TemperatureLogitsProcessor(10.0), TopPLogitsProcessor(top_p=0.5), TopKLogitsProcessor(top_k=1)])

    scores = list_processor(input_ids, scores)
    print(scores)