# NEURAL MACHINE TRANSLATION BY JOINTLY LEARNING TO ALIGN AND TRANSLATE
# https://arxiv.org/pdf/1409.0473

import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytest

class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Va(tanh(Wa[h_enc; h_dec])) from the paper
        self.W_a = nn.Linear(3*hidden_size, hidden_size)  # 3*hidden: concat of decoder state and bidirectional encoder state
        self.v_a = nn.Linear(hidden_size, 1, bias=False)  # vector v_a for computing alignment scores
        
    def forward(self, decoder_hidden, encoder_outputs):
        """
        Args:
            decoder_hidden: Current decoder hidden state (batch_size, hidden_size)
            encoder_outputs: Bidirectional encoder outputs (batch_size, seq_len, 2*hidden_size)
        Returns:
            context: Context vector (batch_size, 2*hidden_size)
            attention_weights: Alignment scores (batch_size, seq_len)
        """
        seq_len = encoder_outputs.size(1)
        decoder_hidden = decoder_hidden.unsqueeze(1).repeat(1, seq_len, 1)
        
        # [batch_size, seq_len, 3*hidden_size]
        concat = torch.cat((encoder_outputs, decoder_hidden), dim=2)
        
        # e_ij = v_a^T * tanh(W_a[h_enc; h_dec])
        energy = torch.tanh(self.W_a(concat))
        alignment = self.v_a(energy)
        
        # alpha_ij = exp(e_ij) / sum_k exp(e_ik)
        attention_weights = F.softmax(alignment.squeeze(-1), dim=1)
        
        # c_i = sum_j alpha_ij * h_j
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        
        return context, attention_weights
    
def test_bahdanau_attention():
    batch_size = 32
    seq_len = 10
    hidden_size = 256
    
    attention = BahdanauAttention(hidden_size)
    
    decoder_hidden = torch.randn(batch_size, hidden_size)
    encoder_outputs = torch.randn(batch_size, seq_len, 2*hidden_size)
    
    context, attention_weights = attention(decoder_hidden, encoder_outputs)
    
    assert context.shape == (batch_size, 2*hidden_size)
    assert attention_weights.shape == (batch_size, seq_len)
    
    assert torch.allclose(attention_weights.sum(dim=1), torch.ones(batch_size), atol=1e-6)
    assert torch.all(attention_weights >= 0)
    assert torch.all(attention_weights <= 1)

    print("Test passed")

if __name__ == "__main__":
    test_bahdanau_attention()

    