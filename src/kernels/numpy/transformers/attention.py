import numpy as np

"""
Attention:

Attention is a mechanism that allows each position in a sequence to attend to all other
positions in the sequence. This allows the model to capture global dependencies in the sequence.
In other words, there are no constraints on distance for building relationships between parts of the
sequence.

Algorithm:
    1. Obtain attention scores by computing dot product of the Query and Key matrices. (Q * K^T)
    2. Apply a scaling factor to the attention scores. (sqrt(d_k))
    3. Apply a softmax function to normalize attention scores and obtain attention weights. (softmax('scaled attention scores'))
    4. Compute the weighted sum of the Value matrix using the attention weights. (attn_weights * V)

Formula:
    Let Q be the Query matrix, K be the Key matrix, V be the Value matrix, and d_k be the dimension of the Key.
    Attention(Q, K, V) = softmax((Q * K^T) / sqrt(d_k)) @ V
"""
def attention(
    Q: np.ndarray, 
    K: np.ndarray, 
    V: np.ndarray, 
    d_k: float
) -> np.ndarray:
    attention_scores = np.dot(Q, K.T)
    scaled_attention_scores = attention_scores / np.sqrt(d_k)
    attention_weights = (np.exp(scaled_attention_scores - np.max(scaled_attention_scores)) / 
                         np.exp(scaled_attention_scores - np.max(scaled_attention_scores)).sum(axis=0)) # softmax(scaled_attention_scores)
    return attention_weights @ V
    
