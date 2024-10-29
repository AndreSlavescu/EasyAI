"""
Dense Passage Retrieval

Paper:
    https://arxiv.org/pdf/2004.04906
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer

# for retrieval example
import faiss

class SimpleEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_dim, max_length):
        super(SimpleEmbedding, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_length, embed_dim)
        
    def forward(self, input_ids):
        positions = torch.arange(0, input_ids.size(1), device=input_ids.device).unsqueeze(0)
        return self.token_embedding(input_ids) + self.position_embedding(positions)

class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(SelfAttention, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads)
        
    def forward(self, x):
        x = x.permute(1, 0, 2)  # (seq_len, batch, embed_dim)
        attn_output, _ = self.attention(x, x, x)
        return attn_output.permute(1, 0, 2)

class TransformerLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden_dim):
        super(TransformerLayer, self).__init__()
        self.self_attention = SelfAttention(embed_dim, num_heads)
        self.linear1 = nn.Linear(embed_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, embed_dim)
        
    def forward(self, x):
        attn_output = self.self_attention(x)
        x = x + attn_output
        forward_output = F.relu(self.linear1(x))
        forward_output = self.linear2(forward_output)
        return x + forward_output

class DPRModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, max_length=128, num_heads=4, hidden_dim=256):
        super(DPRModel, self).__init__()
        self.embedding = SimpleEmbedding(vocab_size, embed_dim, max_length)
        self.transformer = TransformerLayer(embed_dim, num_heads, hidden_dim)
        self.pooling = nn.AdaptiveAvgPool1d(1)
        
    def forward(self, input_ids):
        x = self.embedding(input_ids)
        x = self.transformer(x)
        x = x.mean(dim=1)
        return x

def dpr_loss(q_embeddings, p_embeddings, temperature=1.0):
    similarity_scores = torch.matmul(q_embeddings, p_embeddings.T) / temperature
    labels = torch.arange(q_embeddings.size(0), device=q_embeddings.device)
    return F.cross_entropy(similarity_scores, labels)

def prepare_data(tokenizer, texts, max_length=128):
    tokens = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors='pt')
    return tokens['input_ids']

def train(epochs: int = 100, learning_rate = 1e-4):
    optimizer = torch.optim.Adam(list(question_encoder.parameters()) + list(passage_encoder.parameters()), lr=learning_rate)

    for epoch in range(epochs):
        optimizer.zero_grad()
        
        q_embeddings = question_encoder(question_input_ids)
        p_embeddings = passage_encoder(passage_input_ids)
        
        loss = dpr_loss(q_embeddings, p_embeddings)
        loss.backward()
        optimizer.step()
        
        print(f"Epoch {epoch + 1}, Loss: {loss.item()}")
        
def retrieve(index, tokenizer, query, top_k=2):
    query_ids = prepare_data(tokenizer, [query])
    with torch.no_grad():
        query_embedding = question_encoder(query_ids).cpu().numpy()
    _, indices = index.search(query_embedding, top_k)
    return [passages[i] for i in indices[0]]


if __name__ == "__main__":
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    vocab_size = tokenizer.vocab_size

    question_encoder = DPRModel(vocab_size=vocab_size)
    passage_encoder = DPRModel(vocab_size=vocab_size)

    questions = ["Who is the 44th president of the United States?", "Who has the most points in the NBA?"]
    passages = ["Barack Obama.", "LeBron James."]

    question_input_ids = prepare_data(tokenizer, questions)
    passage_input_ids = prepare_data(tokenizer, passages)
    
    print("STARTING TRAINING!\n\n")
    train()
    print("\n\nFINISHED TRAINING!")

    with torch.no_grad():
        passage_embeddings = passage_encoder(passage_input_ids).cpu().numpy()

    index = faiss.IndexFlatIP(passage_embeddings.shape[1]) # inner product based similarity
    index.add(passage_embeddings)
    
    query = "Who has the most points in the NBA?"
    print(f"\n\nQuery: {query}:\n\nRetrieved Passage: {retrieve(index, tokenizer, query)[0]}")


