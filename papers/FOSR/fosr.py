# FOSR: FIRST-ORDER SPECTRAL REWIRING FOR ADDRESSING OVERSQUASHING IN GNNS
# https://openreview.net/pdf?id=3YjQfCLdrzz

import torch
import torch.nn as nn
from utils import Graph

class FOSR(nn.Module):
    """
    input: 
        graph: Graph G
        k: int - iteration count
        r: float - initial number of power iterations
    """
    def __init__(self, graph: Graph, k: int, r: float):
        super().__init__()
        self.graph = graph
        self.k = k
        self.r = r

    def power_iteration_with_normalization(self, x: torch.Tensor) -> torch.Tensor:
        x = x.reshape(-1, 1)
        
        # adjacency matrix
        A = self.graph.get_adjacency_matrix()
        
        # degree of each node
        d = A.sum(dim=1).reshape(-1, 1)
        
        eps = 1e-6
        D = self.graph.get_degree_matrix() + eps * torch.eye(self.graph.num_vertices)
        
        # inner product
        inner_product = (x * torch.sqrt(d + eps)).sum()

        # num edges
        m = int(A.sum() / 2)
        if m == 0: 
            m = 1

        x = D.inverse().sqrt() @ A @ D.inverse().sqrt() @ x - inner_product / (2 * m) * torch.sqrt(d + eps)
        x = x / (x.norm() + eps)

        return x.flatten()

    def add_edge(self, x: torch.Tensor):
        d = self.graph.get_degree_vector()
        best_score = -float('inf')
        best_edge = None

        # add edge (i, j) which minimizes (x_i * x_j) / sqrt((1 + d_i)(1 + d_j))
        for i in range(self.graph.num_vertices):
            for j in range(i + 1, self.graph.num_vertices):
                if not self.graph.has_edge(i, j):
                    score = (x[i] * x[j]) / torch.sqrt((1 + d[i]) * (1 + d[j]))
                    if score > best_score:
                        best_score = score
                        best_edge = (i, j)
        
        self.graph.add_edge(best_edge)
        return

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for _ in range(self.r):
            x = self.power_iteration_with_normalization(x)

        for _ in range(self.k):
            self.add_edge(x)
            x = self.power_iteration_with_normalization(x)
        return x

def generate_edges(num_vertices: int, num_edges: int) -> torch.Tensor:
    edges = torch.zeros((num_edges, 2), dtype=torch.long)
    for i in range(num_edges):
        src = torch.randint(0, num_vertices, (1,)).item()
        dst = torch.randint(0, num_vertices, (1,)).item()
        while dst == src:
            dst = torch.randint(0, num_vertices, (1,)).item()
        edges[i, 0] = src
        edges[i, 1] = dst
    return edges

def main():
    dim = 20 # arbitrary dimension
    x = torch.randn(dim)
    edges = generate_edges(dim, dim)
    graph = Graph(dim, edges)
    fosr = FOSR(graph, dim, dim)
    x = fosr(x)
    print(x)

if __name__ == "__main__":
    main()