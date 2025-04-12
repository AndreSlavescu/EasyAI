import numpy as np
from typing import List

class KVCache:
    def __init__(self, num_layers, num_heads, head_dim):
        """
        Initialize the KVCache for an transformer model.

        Parameters:
        - num_layers: Number of transformer layers.
        - num_heads: Number of attention heads per layer.
        - head_dim: Dimension of each attention head.
        """
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim

        self.keys: List[List[np.ndarray]] = [[] for _ in range(num_layers)] # per layer
        self.vals: List[List[np.ndarray]] = [[] for _ in range(num_layers)] # per layer

    def append(self, layer_idx, key, value):
        """
        Append key and value tensors to the cache for a specific layer.

        Parameters:
        - layer_idx: Index of the transformer layer.
        - key: Key tensor of shape (num_heads, head_dim).
        - value: Value tensor of shape (num_heads, head_dim).
        """
        self.keys[layer_idx].append(key)
        self.vals[layer_idx].append(value)

    def get(self, layer_idx):
        """
        Retrieve the stacked keys and values for a specific layer.

        Parameters:
        - layer_idx: Index of the transformer layer.

        Returns:
        - keys: Stacked key tensors of shape (sequence_length, num_heads, head_dim).
        - values: Stacked value tensors of shape (sequence_length, num_heads, head_dim).

        Implementation Steps:
        - Stack the list of keys and values along the sequence dimension.
        - Return the stacked tensors for use in attention computation.
        """
        key = self.keys[layer_idx]
        val = self.vals[layer_idx]

        if not key:
            return None, None
        
        return np.stack(key), np.stack(val)

    def clear(self):
        """
        Clear the KV cache.

        Implementation Steps:
        - Reset the data structures storing keys and values for all layers.
        - Ensure that the cache is empty and ready for a new sequence.
        """
        self.keys = [[] for _ in range(self.num_layers)]
        self.vals = [[] for _ in range(self.num_layers)]

def test_kv_cache():
    """
    Test the KVCache implementation with a simulated LLM setup.
    """
    num_layers = 2
    num_heads = 4
    head_dim = 64
    sequence_length = 5

    kv_cache = KVCache(num_layers, num_heads, head_dim)

    for t in range(sequence_length):
        for layer in range(num_layers):
            key = np.random.randn(num_heads, head_dim)
            value = np.random.randn(num_heads, head_dim)
            kv_cache.append(layer, key, value)

    for layer in range(num_layers):
        keys, values = kv_cache.get(layer)
        assert keys.shape == (sequence_length, num_heads, head_dim), f"Keys shape mismatch in layer {layer}"
        assert values.shape == (sequence_length, num_heads, head_dim), f"Values shape mismatch in layer {layer}"
        print(f"Layer {layer}: Keys shape {keys.shape}, Values shape {values.shape}")

    kv_cache.clear()
    for layer in range(num_layers):
        keys, values = kv_cache.get(layer)
        assert keys is None and values is None, f"Cache not cleared properly for layer {layer}"
    print("KV cache cleared successfully.")

if __name__ == "__main__":
    test_kv_cache()
