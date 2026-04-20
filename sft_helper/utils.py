import torch
import torch.nn.functional as F
from transformers import PreTrainedModel
def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """
    Compute the entropy of a probability distribution given its logits.

    Parameters
    ----------
    logits : torch.Tensor
        A tensor of shape (batch_size, sequence_length, vocab_size) containing the logits
        for each token in the vocabulary at each position in the sequence.

    Returns
    -------
    torch.Tensor
        A tensor of shape (batch_size, sequence_length) containing the entropy of the
        probability distribution at each position in the sequence.
    """
    # Compute probabilities from logits using softmax
    probabilities = F.softmax(logits, dim=-1)

    # Compute log probabilities
    log_probabilities = F.log_softmax(logits, dim=-1)

    # Compute entropy using the formula: H(p) = -sum(p * log(p))
    entropy = -torch.sum(probabilities * log_probabilities, dim=-1)

    return entropy
def get_response_log_probs( model: PreTrainedModel, input_ids: torch.Tensor, labels: torch.Tensor, return_token_entropy: bool=False)-> dict[str, torch.Tensor]:
    """
    Get the log probabilities of the response tokens in the labels, given the input_ids.

    Parameters
    ----------
    model : PreTrainedModel
        The language model to use for computing logits.
    input_ids : torch.Tensor
        A tensor of shape (batch_size, sequence_length) containing the token IDs for the input sequences.
    labels : torch.Tensor
        A tensor of shape (batch_size, sequence_length) containing the token IDs for the output sequences (labels).
    return_token_entropy : bool, optional
        Whether to return the entropy of the token distributions, by default False.

    Returns
    -------
    dict[str, torch.Tensor]
        A dictionary containing:
        - "log_probs": A tensor of shape (batch_size, sequence_length) containing the log probabilities of the response tokens.
        - "token_entropy" (optional): A tensor of shape (batch_size, sequence_length) containing the entropy of the token distributions at each position.
    """
    # Get logits from the model
    with torch.no_grad():
        outputs = model(input_ids=input_ids)
        logits = outputs.logits

    # Compute log probabilities from logits
    log_probs = F.log_softmax(logits, dim=-1)

    # Create a mask for response tokens (where labels are not equal to -100)
    response_mask = labels != -100

    # Gather log probabilities for response tokens
    response_log_probs = torch.zeros_like(labels, dtype=torch.float)
    response_log_probs[response_mask] = log_probs[response_mask.unsqueeze(-1).expand_as(log_probs)].gather(-1, labels[response_mask].unsqueeze(-1)).squeeze(-1)

    result = {"log_probs": response_log_probs}

    if return_token_entropy:
        token_entropy = compute_entropy(logits)
        result["token_entropy"] = token_entropy

    return result
