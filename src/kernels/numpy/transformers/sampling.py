import numpy as np

"""
Top-p (nucleus) sampling:

Formula:
    Let P be the set of probabilities of the next token
    Let p be the cumulative probability threshold.
    Find the smallest k such that:
        sum(P[:k]) >= p
    Select the tokens corresponding to P[:k].
"""
def top_p_sampling(probabilities: np.ndarray, p_threshold: float):
    sorted_indices = np.argsort(probabilities)[::-1]
    probabilities = probabilities[sorted_indices]
    cumulative_sum = np.cumsum(probabilities)
    p = np.searchsorted(cumulative_sum, p_threshold) + 1
    return probabilities[:p]
    

"""
Top-k sampling:

Formula:
    Let P be the set of probabilities of the next token
    Let k be the number of top tokens to consider.
    Select the tokens corresponding to the top k probabilities.
"""
def top_k_sampling(probabilities: np.ndarray, k: float):
    sorted_indices = np.argsort(probabilities)[::-1]
    probabilities = probabilities[sorted_indices]
    k = np.searchsorted(probabilities, k) + 1
    return probabilities[:k]


"""
Gumbel Softmax Sampling:

Formula:
    Let G be the Gumbel distribution.
    Sample g_i from G for each logit l_i.
    Compute y_i = softmax((l_i + g_i) / temperature).
    Select the token with the highest value of y_i.
"""
def gumbel_softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    gumbel_noise = -np.log(-np.log(np.random.uniform(0, 1, logits.shape)))
    y = logits + gumbel_noise
    return np.exp(y / temperature) / np.sum(np.exp(y / temperature), axis=-1, keepdims=True)

