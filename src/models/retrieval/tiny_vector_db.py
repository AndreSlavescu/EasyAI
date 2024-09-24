from datasets import load_dataset
import numpy as np

from dataclasses import dataclass
from typing import Optional, Dict, List
import random

@dataclass
class Index:
    data: Dict[str, np.ndarray]
    size: int

class VectorDB:
    EMBEDDING_SIZE = 128

    def __init__(self, index: Optional[Index] = None):
        if index:
            self.db = self.embed_index(index)
        else:
            self.db = Index(data={}, size=0)

    # simple custom embedding function, this can be replaced 
    #  with another embedding function if desired
    @staticmethod
    def embed(value):
        if isinstance(value, str):
            ascii_values = np.fromiter((ord(char) for char in value if 0 <= ord(char) <= 255), dtype=np.uint8)
            normalized_values = np.array(ascii_values) / 255.0
            if len(normalized_values) < VectorDB.EMBEDDING_SIZE:
                padded_values = np.pad(normalized_values, (0, VectorDB.EMBEDDING_SIZE - len(normalized_values)), 'constant')
            else:
                padded_values = normalized_values[:VectorDB.EMBEDDING_SIZE]
            return padded_values
        else:
            raise ValueError("Unsupported data type for embedding")
        
    @staticmethod
    def embed_index(index: Index) -> Index:
        embedded_data = {key: VectorDB.embed(value) for key, value in index.data.items()}
        return Index(data=embedded_data, size=index.size)

    def add_vector(self, key: str, value: str):
        embedded_value = self.embed(value)
        self.db.data[key] = embedded_value
        self.db.size += 1

    def get_vector(self, key: str) -> Optional[np.ndarray]:
        if key in self.db.data:
            return self.db.data.get(key)
        raise ValueError("Key not in VectorDB")
    
    def delete_vector(self, key: str):
        if key in self.db.data:
            del self.db.data[key]
            self.db.size -= 1

    def search(self, query: str, top_k: int = 5) -> List[str]:
        query_vector = self.embed(query).reshape(1, -1)
        data_vectors = np.array(list(self.db.data.values()))
        keys = list(self.db.data.keys())
        distances = np.linalg.norm(data_vectors - query_vector, axis=1)
        top_k_indices = np.argsort(distances)[:top_k]
        top_k_results = [keys[i] for i in top_k_indices]
        return top_k_results

def main():
    msmarco_data = load_dataset('ms_marco', 'v2.1', split='train')
    initial_data = {msmarco_data[i]['query']: msmarco_data[i]['passages']['passage_text'][0] for i in random.sample(range(len(msmarco_data)), 3)}
    
    index = Index(data=initial_data, size=len(initial_data))
    vector_db = VectorDB(index)
    
    random_index = random.randint(0, len(msmarco_data) - 1)
    random_query = msmarco_data[random_index]['query']
    random_passage = msmarco_data[random_index]['passages']['passage_text'][0]
    vector_db.add_vector(random_query, random_passage)

    try:
        random_query_for_retrieval = random.choice(list(initial_data.keys()))
        vector = vector_db.get_vector(random_query_for_retrieval)
        print(f"Retrieved vector for '{random_query_for_retrieval}':", vector)
    except ValueError as e:
        print(e)

    random_query_for_deletion = random.choice(list(initial_data.keys()))
    vector_db.delete_vector(random_query_for_deletion)
    print(f"\n\nDeleted vector for '{random_query_for_deletion}'")

    random_query_for_search = msmarco_data[random_index]['query']
    search_results = vector_db.search(random_query_for_search, top_k=2)
    print(f"\n\nSearch results for '{random_query_for_search}':", search_results)

if __name__ == "__main__":
    main()