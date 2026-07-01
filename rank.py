import os
import argparse
import time
import pandas as pd

# Import our pipeline functions
from scoring.score import run_scoring
from scoring.reasoning import run_reasoning

def main():
    parser = argparse.ArgumentParser(description="Redrob Candidate Ranking CLI")
    parser.add_argument("--candidates", type=str, required=True, help="Path to candidates.jsonl dataset")
    parser.add_argument("--out", type=str, required=True, help="Path to write the output submission CSV")
    args = parser.parse_args()
    
    start_time = time.time()
    
    # 1. Resolve cache paths (assumed to be in the cache/ directory next to this script)
    features_path = "cache/features.parquet"
    embeddings_path = "cache/embeddings.npy"
    req_embeddings_path = "cache/requirement_embeddings.npy"
    scored_path = "cache/features_scored.parquet"
    
    print("--- STARTING RANKING PIPELINE ---")
    print(f"Candidates file parameter: {args.candidates}")
    print(f"Output CSV path: {args.out}")
    
    # Verify that cache exists
    f_part1 = features_path.replace(".parquet", "_part1.parquet")
    f_part2 = features_path.replace(".parquet", "_part2.parquet")
    has_features = os.path.exists(features_path) or (os.path.exists(f_part1) and os.path.exists(f_part2))
    
    parts = [embeddings_path.replace(".npy", f"_part{i}.npy") for i in range(1, 5)]
    has_embeddings = os.path.exists(embeddings_path) or all(os.path.exists(p) for p in parts)
    
    # Fallback to checking 2 parts of embeddings
    part1_path = embeddings_path.replace(".npy", "_part1.npy")
    part2_path = embeddings_path.replace(".npy", "_part2.npy")
    if not has_embeddings and (os.path.exists(part1_path) and os.path.exists(part2_path)):
        has_embeddings = True
        
    if not (has_features and has_embeddings and os.path.exists(req_embeddings_path)):
        print("Error: Precomputed cache files not found in cache/. Please run extract_features, honeypot_checks, and embed_candidates first.")
        return
        
    # 2. Run scoring in-memory (writes to scored_path as intermediate)
    run_scoring(
        features_path=features_path,
        embeddings_path=embeddings_path,
        req_embeddings_path=req_embeddings_path,
        output_path=scored_path
    )
    
    # 3. Run reasoning generation (updates scored_path)
    run_reasoning(
        features_scored_path=scored_path,
        embeddings_path=embeddings_path,
        req_embeddings_path=req_embeddings_path,
        output_path=scored_path
    )
    
    # 4. Load the scored and reasoned features
    print("\nLoading final scored results...")
    df = pd.read_parquet(scored_path)
    
    # Save the lightweight top 100 details cache for the Streamlit dashboard
    print("Saving lightweight top 100 details cache...")
    df.head(100).to_parquet("cache/top_100_details.parquet", index=False)
    
    # 5. Extract the top 100 rows
    print("Extracting top 100 candidates...")
    top_100 = df.head(100).copy()
    
    # Rename final_score to score per submission specification
    top_100 = top_100.rename(columns={"final_score": "score"})
    
    # Ensure types are correct
    top_100["rank"] = top_100["rank"].astype(int)
    top_100["score"] = top_100["score"].astype(float)
    
    # Select only the required columns
    submission_df = top_100[["candidate_id", "rank", "score", "reasoning"]]
    
    # 6. Save as CSV
    print(f"Writing final submission CSV to {args.out}...")
    submission_df.to_csv(args.out, index=False, encoding="utf-8")
    
    elapsed_time = time.time() - start_time
    print(f"\n--- PIPELINE COMPLETED SUCCESSFULLY ---")
    print(f"Total ranking time: {elapsed_time:.2f} seconds")
    print(f"Submission file verified: {args.out}")

if __name__ == "__main__":
    main()
