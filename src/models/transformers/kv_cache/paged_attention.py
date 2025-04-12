import torch
from typing import List

class PagedAttentionCache:
    def __init__(self, num_heads, head_dim, page_size, max_pages):
        """
        Initialize the cache system with preallocated pages.
        """
        self.num_heads: int = num_heads
        self.head_dim: int = head_dim
        self.page_size: int = page_size
        self.max_pages: int = max_pages

        self.page_table: List[dict] = [{} for _ in range(max_pages)]
        self.keys: torch.Tensor = torch.zeros(max_pages, page_size, num_heads, head_dim)
        self.vals: torch.Tensor = torch.zeros_like(self.keys)
        self.next_free_page: int = 0

    def _get_next_free_page(self):
        quotient, remainder = divmod(self.next_free_page + 1, self.max_pages)
        if quotient >= 1:
            raise RuntimeError("No more free pages left. Failing insertion.")

        self.next_free_page = remainder

    def append_kv(self, batch_id, token_ids, key_block, value_block):
        """
        Append key/value tensors to the cache.
        Each token in token_ids is assigned a page.
        """
        if self.next_free_page + len(token_ids) > self.max_pages:
            raise RuntimeError(f"KV Cache overflow: need {len(token_ids)} pages but only {self.max_pages - self.next_free_page} available")

        for i, token_id in enumerate(token_ids):
            self.page_table[self.next_free_page][batch_id] = token_id
            self.keys[self.next_free_page, 0] = key_block[i]
            self.vals[self.next_free_page, 0] = value_block[i]
            self._get_next_free_page()

    def gather_kv(self, batch_id, seq_len):
        """
        Gather all KV pairs for tokens in a sequence.
        Returns:
            gathered_keys: (seq_len, num_heads, head_dim)
            gathered_values: (seq_len, num_heads, head_dim)
        """
        gathered_keys = torch.zeros(seq_len, self.num_heads, self.head_dim)
        gathered_vals = torch.zeros_like(gathered_keys)

        for page_idx, batch_dict in enumerate(self.page_table):
            if batch_id in batch_dict:
                token_id = batch_dict[batch_id]
                if token_id < seq_len:
                    gathered_keys[token_id] = self.keys[page_idx, 0]
                    gathered_vals[token_id] = self.vals[page_idx, 0]

        return gathered_keys, gathered_vals

def test_paged_attention_basic():
    print("Running test: Basic Append & Gather")

    batch_id = 0
    num_heads = 2
    head_dim = 4
    page_size = 1
    max_pages = 8
    seq_len = 4

    cache = PagedAttentionCache(num_heads, head_dim, page_size, max_pages)

    key_block = torch.randn(seq_len, num_heads, head_dim)
    value_block = torch.randn(seq_len, num_heads, head_dim)
    token_ids = list(range(seq_len))

    cache.append_kv(batch_id, token_ids, key_block, value_block)

    gathered_k, gathered_v = cache.gather_kv(batch_id, seq_len)

    assert torch.allclose(gathered_k, key_block, atol=1e-5), "Key mismatch"
    assert torch.allclose(gathered_v, value_block, atol=1e-5), "Value mismatch"

    print("Passed!")


def test_paged_attention_multiple_batches():
    print("Running test: Multiple Batch Isolation")

    num_heads = 2
    head_dim = 4
    page_size = 1
    max_pages = 16
    seq_len = 3

    cache = PagedAttentionCache(num_heads, head_dim, page_size, max_pages)

    for batch_id in [0, 1]:
        key_block = torch.randn(seq_len, num_heads, head_dim)
        value_block = torch.randn(seq_len, num_heads, head_dim)
        token_ids = list(range(seq_len))

        cache.append_kv(batch_id, token_ids, key_block, value_block)
        gathered_k, gathered_v = cache.gather_kv(batch_id, seq_len)

        assert torch.allclose(gathered_k, key_block, atol=1e-5), f"Key mismatch for batch {batch_id}"
        assert torch.allclose(gathered_v, value_block, atol=1e-5), f"Value mismatch for batch {batch_id}"

    print("Passed!")


def test_paged_attention_overflow():
    print("Running test: Memory Overflow Handling")

    num_heads = 1
    head_dim = 2
    page_size = 1
    max_pages = 2
    seq_len = 3

    cache = PagedAttentionCache(num_heads, head_dim, page_size, max_pages)

    try:
        key_block = torch.randn(seq_len, num_heads, head_dim)
        value_block = torch.randn(seq_len, num_heads, head_dim)
        token_ids = list(range(seq_len))
        cache.append_kv(0, token_ids, key_block, value_block)
        print("Failed: Expected overflow but didn't get one.")
    except RuntimeError:
        print("Passed (Overflow correctly handled)")



if __name__ == "__main__":
    test_paged_attention_basic()
    test_paged_attention_multiple_batches()
    test_paged_attention_overflow()
