import os
import pickle
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss


class VectorDatabase:
    """
    A class for encoding documents and retrieving relevant chunks using FAISS.
    Stores model, index, and chunks in memory for efficient retrieval.
    """
    
    def __init__(self, bi_encoder: str = 'sentence-transformers/multi-qa-MiniLM-L6-cos-v1', cross_encoder: str = 'cross-encoder/ms-marco-MiniLM-L6-v2'):
        """
        Initialize the VectorDatabase with a SentenceTransformer model.
        
        Args:
            bi_encoder/cross_encoder: HuggingFace name of the model
        """
        self.bi_encoder = SentenceTransformer(bi_encoder)
        self.cross_encoder = CrossEncoder(cross_encoder)
        self.index = None
        self.chunks = None


    def encode_document(self, doc_path: str, save_path: str, verbose: bool = True):
        """
        Encode a document into vector embeddings and create a FAISS index.
        
        Args:
            doc_path: Path to document
            save_path: Optional path to save the index and chunks to disk
        """
        # Read markdown file
        with open(doc_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split on "###" and remove empty strings
        self.chunks = [chunk.strip() for chunk in content.split('###') if chunk.strip()]
        
        # Embed chunks
        doc_emb = self.bi_encoder.encode(self.chunks)  # Numpy (float32), (chunks, emb_dim)
        if verbose:
            lengths = [len(chunk) for chunk in self.chunks]
            print(f"Created {len(self.chunks)} embeddings with dimension {doc_emb.shape[1]}")
            print(f"Max document length: {max(lengths)}")
            print(f"Avg document length: {sum(lengths) / len(lengths):.2f}")
        
        # Create FAISS index
        self.index = faiss.IndexFlatL2(doc_emb.shape[1])  # L2 distance
        
        # Add embeddings to index
        self.index.add(doc_emb.astype('float32'))
        
        # Save to disk
        if self.index is None or self.chunks is None:
            raise ValueError("No index or chunks to save. Encode a document first.")
        
        # Create output directory if it doesn't exist
        os.makedirs(save_path, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, os.path.join(save_path, 'faiss_index.bin'))
        
        # Save chunks
        with open(os.path.join(save_path, 'chunks.pkl'), 'wb') as f:
            pickle.dump(self.chunks, f)
        
        if verbose:
            print(f"Saved FAISS index and chunks to '{save_path}'")
    

    def load_or_create(self, doc_path: str, save_path: str = 'vector_db', verbose: bool = True):
        """
        Load existing vector database if it exists, otherwise create and save a new one.
        
        Args:
            doc_path: Path to the markdown document (used if creating new database)
            save_path: Path to the vector database directory
        """
        index_file = os.path.join(save_path, 'faiss_index.bin')
        chunks_file = os.path.join(save_path, 'chunks.pkl')
        
        # Check if both files exist
        if os.path.exists(index_file) and os.path.exists(chunks_file):
            if verbose:
                print(f"Loading existing vector database from '{save_path}'...")
            self.load(save_path, verbose=verbose)
        else:
            if verbose:
                print(f"Creating new vector database from '{doc_path}'...")
            self.encode_document(doc_path, save_path=save_path, verbose=verbose)
    

    def load(self, load_path: str, verbose: bool = True):
        """
        Load a previously saved FAISS index and chunks from disk.
        
        Args:
            load_path: Path to the directory containing the index and chunks
        """
        # Load FAISS index
        self.index = faiss.read_index(os.path.join(load_path, 'faiss_index.bin'))
        
        # Load chunks
        with open(os.path.join(load_path, 'chunks.pkl'), 'rb') as f:
            self.chunks = pickle.load(f)
        
        if verbose:
            print(f"Loaded index with {self.index.ntotal} vectors and {len(self.chunks)} chunks")
    

    def retrieve(self, query: str, top_k: int = 30, max_chars: int = 4000):
        """
        Retrieve top-k most relevant document chunks given a query from the bi-encoder and
        pick top-n chunks starting from the largest difference from the cross-encoder.
        
        Args:
            query: User's question/query
            top_k: Number of top documents to retrieve from bi-encoder
            max_chars: Maximum total characters to return
        
        Returns:
            String containing retrieved chunks joined by newlines
        """
        if self.index is None or self.chunks is None:
            raise ValueError("No index loaded. Encode or load a document first.")
        
        # Encode query
        query_emb = self.bi_encoder.encode([query])  # Shape: (1, embedding_dim)
        
        # Search for top-k nearest neighbors (Bi-encoder)
        _, indices = self.index.search(query_emb.astype('float32'), top_k)
        initial_chunks = [self.chunks[idx] for idx in indices[0]]

        # Cross-encoder reranking
        pairs = [(query, chunk) for chunk in initial_chunks]
        cross_scores = self.cross_encoder.predict(pairs)

        # Create list of (score, index) tuples and sort
        scored_chunks = list(zip(cross_scores, indices[0])) # indices: [[idx_0, ..., idx_n]]
        scored_chunks.sort(key=lambda x: x[0], reverse=True)

        # Find the LARGEST score gap
        max_gap = 0
        max_gap_position = len(scored_chunks)  # Default: no cutoff
        for i in range(len(scored_chunks) - 1):
            score_diff = scored_chunks[i][0] - scored_chunks[i + 1][0]
    
            if score_diff > max_gap:
                max_gap = score_diff
                max_gap_position = i + 1
        relevant_chunks = scored_chunks[:max_gap_position] # Cut irrelevant chunks
        
        # Apply character limit (no fragmentation)
        retrieved_chunks = []
        total_chars = 0
        for _, idx in relevant_chunks:
            chunk = self.chunks[idx]
            chunk_length = len(chunk)
            
            if total_chars + chunk_length <= max_chars:
                retrieved_chunks.append(chunk)
                total_chars += chunk_length
        
        return "\n\n".join(retrieved_chunks)
