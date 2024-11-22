# Graph Tools
import torch

class Graph:
    def __init__(self, num_vertices: int, edges: torch.Tensor):
        assert edges.shape[1] == 2
        self.num_vertices = num_vertices
        self.edges = edges.t()

    def get_adjacency_matrix(self) -> torch.Tensor:
        adj_matrix = torch.zeros(self.num_vertices, self.num_vertices)
        adj_matrix[self.edges[0], self.edges[1]] = 1
        adj_matrix[self.edges[1], self.edges[0]] = 1
        return adj_matrix

    def get_degree_matrix(self) -> torch.Tensor:
        degree_matrix = torch.zeros(self.num_vertices, self.num_vertices)
        degree_matrix[torch.arange(self.num_vertices), torch.arange(self.num_vertices)] = self.get_adjacency_matrix().sum(dim=1)
        return degree_matrix

    def get_degree_vector(self) -> torch.Tensor:
        return self.get_degree_matrix().sum(dim=1)

    def has_edge(self, i: int, j: int) -> bool:
        return self.get_adjacency_matrix()[i, j] == 1 or self.get_adjacency_matrix()[j, i] == 1

    def add_edge(self, edge: tuple[int, int]):
        new_edge = torch.tensor([[edge[0]], [edge[1]]])
        self.edges = torch.cat([self.edges, new_edge], dim=1)

def main():
    vertices = 10
    edges = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 0]])
    graph = Graph(vertices, edges)
    print("Adjacency Matrix")
    print(graph.get_adjacency_matrix())
    print("Degree Matrix")
    print(graph.get_degree_matrix())
    print("Degree Vector")
    print(graph.get_degree_vector())
    print("Has Edge (0, 1)")
    print(graph.has_edge(0, 1))
    print("Has Edge (1, 0)")
    print(graph.has_edge(1, 0))
    print("Add Edge (0, 9)")
    graph.add_edge((0, 9))
    edges = graph.edges.t()
    print(edges)

if __name__ == "__main__":
    main()
