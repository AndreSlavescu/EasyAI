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


"""
Multi-Head Attention:

Multi-Head Attention is an extension of the attention mechanism that allows the model to jointly attend to information
from different representation subspaces at different positions. Instead of performing a single attention function, 
multi-head attention runs through the scaled dot-product attention multiple times in parallel, each with different 
learned linear projections of the queries, keys, and values.

Algorithm:
    1. Linearly project the queries, keys, and values h times to obtain h different sets of queries, keys, and values.
    2. Apply the attention function to each set of queries, keys, and values in parallel.
    3. Concatenate the outputs of the attention function and linearly project the concatenated output.

Formula:
    Let Q be the Query matrix, K be the Key matrix, V be the Value matrix, and d_k be the dimension of the Key.
    Let h be the number of heads, and W_Qi, W_Ki, W_Vi be the learned projection matrices for the i-th head.
    MultiHead(Q, K, V) = Concat(head_1, ..., head_h) * W_O
    where head_i = Attention(Q * W_Qi, K * W_Ki, V * W_Vi)
"""
def multihead_attention(
    Q: np.ndarray, 
    K: np.ndarray, 
    V: np.ndarray, 
    d_k: float,
    h: int
) -> np.ndarray:
    d_model = Q.shape[-1]
    assert d_model % h == 0, "d_model must be divisible by the number of heads"

    W_Q = np.random.rand(h, d_model, d_k)
    W_K = np.random.rand(h, d_model, d_k)
    W_V = np.random.rand(h, d_model, d_k)
    W_O = np.random.rand(h * d_k, d_model)

    heads = []
    for i in range(h):
        Q_i = Q @ W_Q[i]
        K_i = K @ W_K[i]
        V_i = V @ W_V[i]
        
        head_i = attention(Q_i, K_i, V_i, d_k)
        heads.append(head_i)

    concatenated_heads = np.concatenate(heads, axis=-1)
    output = concatenated_heads @ W_O
    return output


"""
Linear Attention:

A more efficient variant of attention that reduces the quadratic memory complexity to linear.
Instead of computing the full attention matrix, it uses the associative property of matrix multiplication
to compute attention in linear time and memory.

Formula:
    Let Q be the Query matrix, K be the Key matrix, V be the Value matrix
    LinearAttention(Q,K,V) = ϕ(Q) @ (ϕ(K)^T @ V) / (ϕ(Q) @ ϕ(K)^T @ 1)
    where ϕ is a feature map (here using elu(x) + 1)

This provides O(N) complexity instead of O(N^2) of standard attention.
"""
def linear_attention(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    eps: float = 1e-6
) -> np.ndarray:
    def phi(x):
        return np.maximum(0, x) + np.minimum(0, np.exp(x) - 1) + 1
    
    Q_feat = phi(Q) 
    K_feat = phi(K)

    KV = np.einsum('bsd,bsv->bdv', K_feat, V)
    Z = np.einsum('bsd,bs->bd', K_feat, np.ones_like(K_feat[:,:,0]))
    attention = np.einsum('bqd,bdv->bqv', Q_feat, KV)
    normalizer = np.einsum('bqd,bd->bq', Q_feat, Z)
    output = attention / (normalizer[..., None] + eps)
    
    return output

if __name__ == "__main__":
    batch_size, seq_len, d_model = 2, 4, 8
    num_heads = 2
    d_k = d_model // num_heads

    Q = np.random.randn(batch_size, seq_len, d_model)
    K = np.random.randn(batch_size, seq_len, d_model)
    V = np.random.randn(batch_size, seq_len, d_model)

    expected_shape = (batch_size, seq_len, d_model)

    mha_output = multihead_attention(Q, K, V, d_model, num_heads)
    la_output = linear_attention(Q, K, V)

    np.testing.assert_equal(mha_output.shape, expected_shape)
    np.testing.assert_equal(la_output.shape, expected_shape)
    
    np.testing.assert_array_less(np.abs(mha_output), np.inf)
    np.testing.assert_array_less(np.abs(la_output), np.inf)
    
    np.testing.assert_(not np.isnan(mha_output).any())
    np.testing.assert_(not np.isnan(la_output).any())

    print("All tests passed successfully!")



