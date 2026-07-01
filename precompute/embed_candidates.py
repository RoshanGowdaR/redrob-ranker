import os
import time
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from huggingface_hub import constants

# Set environment variable to disable symlink warnings
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Optimize PyTorch CPU inference
torch.set_num_threads(8)

# Define JD requirements to embed
JD_REQUIREMENTS = [
    "Built and deployed production embeddings-based retrieval systems using sentence-transformers, OpenAI embeddings, BGE, or E5 models, handling embedding drift and index refresh.",
    "Operated vector databases or hybrid search infrastructure such as Pinecone, Weaviate, Qdrant, Milvus, OpenSearch, Elasticsearch, or FAISS in production.",
    "Wrote production-quality Python code as a core part of the job.",
    "Designed and used evaluation frameworks for ranking systems, including NDCG, MRR, MAP, or offline-to-online correlation, and interpreted A/B tests.",
    "Shipped an end-to-end ranking, search, or recommendation system to real users at meaningful scale.",
    "Has LLM fine-tuning experience using LoRA, QLoRA, or PEFT.",
    "Has experience with learning-to-rank models such as XGBoost-based or neural rankers."
]

def generate_embeddings(features_path, embeddings_out, req_out):
    print("Loading features...")
    df = pd.read_parquet(features_path)
    texts = df["career_history_text"].fillna("").tolist()
    
    print("Initializing SentenceTransformer Model (all-MiniLM-L6-v2)...")
    # This will download the model to the local cache if not already present
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    # Identify cache directory
    cache_dir = constants.HF_HUB_CACHE
    print(f"Model is cached locally at: {cache_dir}")
    print(f"Using {torch.get_num_threads()} CPU threads for PyTorch.")
    
    # 1. Embed requirements
    print(f"Embedding {len(JD_REQUIREMENTS)} JD requirements...")
    req_embeddings = model.encode(JD_REQUIREMENTS, show_progress_bar=False, convert_to_numpy=True)
    print(f"Requirement embeddings shape: {req_embeddings.shape}")
    np.save(req_out, req_embeddings)
    print(f"Saved requirement embeddings to {req_out}")
    
    # 2. Embed 100,000 candidates
    print(f"Embedding {len(texts)} candidate career history texts on CPU...")
    start_time = time.time()
    
    # Using batch size of 256 for optimal CPU performance
    candidate_embeddings = model.encode(
        texts,
        batch_size=256,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print(f"\nEmbedding completed successfully!")
    print(f"Candidate embeddings shape: {candidate_embeddings.shape}")
    print(f"Total wall-clock time: {elapsed_time:.2f} seconds ({elapsed_time / 60:.2f} minutes)")
    print(f"Average speed: {len(texts) / elapsed_time:.2f} profiles/sec")
    
    np.save(embeddings_out.replace(".npy", "_part1.npy"), candidate_embeddings[:50000])
    np.save(embeddings_out.replace(".npy", "_part2.npy"), candidate_embeddings[50000:])
    print(f"Saved candidate embeddings split to {embeddings_out.replace('.npy', '_part1.npy')} and {embeddings_out.replace('.npy', '_part2.npy')}")

if __name__ == "__main__":
    os.makedirs("cache", exist_ok=True)
    generate_embeddings(
        features_path="cache/features.parquet",
        embeddings_out="cache/embeddings.npy",
        req_out="cache/requirement_embeddings.npy"
    )
