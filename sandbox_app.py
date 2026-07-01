import streamlit as st
import pandas as pd
import json
import os
import numpy as np

# Set page config
st.set_page_config(
    page_title="GenSonicImpact Candidate Ranker Sandbox",
    layout="wide",
    initial_sidebar_state="expanded"
)

# App Title & Description
st.title("GenSonicImpact — Redrob Candidate Ranker Sandbox")
st.markdown("""
This sandbox allows you to upload a sample candidate dataset in JSONL format, rank the candidates against the Senior AI Engineer job description, and view the scores, disqualifier details, and generated reasoning.
""")

# Load database cache
@st.cache_data
def load_scored_cache():
    cache_path = "cache/features_scored.parquet"
    if os.path.exists(cache_path):
        return pd.read_parquet(cache_path)
    return None

df_cache = load_scored_cache()

# Sidebar options
st.sidebar.header("Configuration")
show_honeypots = st.sidebar.checkbox("Show Honeypots", value=False, help="Include candidates flagged as logical honeypots")

# Check if cache is loaded
if df_cache is None:
    st.error("Error: Scored cache not found. Please run the backend precompute and scoring pipeline first.")
else:
    st.sidebar.success("Scored cache loaded successfully (100k candidates).")

    # File uploader
    uploaded_file = st.file_uploader("Upload candidates sample file (JSONL format)", type=["jsonl", "json"])
    
    # Resolve candidates to display
    candidates_list = []
    
    if uploaded_file is not None:
        st.info("Parsing uploaded candidate sample...")
        try:
            for line in uploaded_file:
                line_str = line.decode("utf-8").strip()
                if line_str:
                    cand_data = json.loads(line_str)
                    cid = cand_data.get("candidate_id")
                    if cid:
                        candidates_list.append(cid)
            st.success(f"Parsed {len(candidates_list)} candidate IDs from upload.")
        except Exception as e:
            st.error(f"Error parsing JSONL file: {e}")
    else:
        # Default sample (first 10 candidates from the top scored list)
        st.info("No file uploaded. Showing default top 10 candidates from precomputed scoring.")
        if show_honeypots:
            default_candidates = df_cache.head(10)["candidate_id"].tolist()
        else:
            default_candidates = df_cache[df_cache["honeypot_flag"] == False].head(10)["candidate_id"].tolist()
        candidates_list = default_candidates

    if candidates_list:
        # Filter candidates from cache
        results_df = df_cache[df_cache["candidate_id"].isin(candidates_list)].copy()
        
        # Filter out honeypots if option is off
        if not show_honeypots:
            results_df = results_df[results_df["honeypot_flag"] == False]
            
        # Re-sort descending by score, then ascending by candidate_id to ensure rank is correct
        results_df = results_df.sort_values(
            by=["final_score", "candidate_id"], 
            ascending=[False, True]
        ).reset_index(drop=True)
        
        # Recalculate ranks for the uploaded sample (1 to N)
        results_df["rank"] = results_df.index + 1
        
        # Display summary metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Candidates Ranked", len(results_df))
        col2.metric("Highest Score", f"{results_df['final_score'].max():.4f}" if len(results_df) > 0 else "N/A")
        col3.metric("Average Score", f"{results_df['final_score'].mean():.4f}" if len(results_df) > 0 else "N/A")
        
        # Display interactive results table
        st.subheader("Ranking Results")
        display_df = results_df[[
            "rank", "candidate_id", "final_score", "semantic_match_score", 
            "signal_modifier", "disqualifier_penalty", "reasoning"
        ]].rename(columns={"final_score": "score"})
        
        st.dataframe(display_df, use_container_width=True)
        
        # Create CSV download
        submission_df = display_df[["candidate_id", "rank", "score", "reasoning"]]
        csv_data = submission_df.to_csv(index=False).encode('utf-8')
        
        st.download_button(
            label="Download Submission CSV",
            data=csv_data,
            file_name="submission.csv",
            mime="text/csv",
        )
        
        # Detail view per candidate
        st.subheader("Candidate Detail Inspector")
        selected_cid = st.selectbox("Select Candidate ID to inspect", display_df["candidate_id"].tolist())
        
        if selected_cid:
            detail = results_df[results_df["candidate_id"] == selected_cid].iloc[0]
            
            c1, c2 = st.columns(2)
            
            with c1:
                st.markdown(f"### Profile Summary: {detail['candidate_id']}")
                st.write(f"**Current Title:** {detail['current_title']}")
                st.write(f"**Current Company:** {detail['current_company']} ({detail['current_company_size']})")
                st.write(f"**Current Industry:** {detail['current_industry']}")
                st.write(f"**Location:** {detail['location']}, {detail['country']}")
                st.write(f"**Years of Experience:** {detail['years_of_experience']} years")
                
            with c2:
                st.markdown("### Score Breakdown")
                st.write(f"**Final Score:** `{detail['final_score']:.6f}`")
                st.write(f"**Semantic Similarity Score:** `{detail['semantic_match_score']:.6f}`")
                st.write(f"**Signal Modifier:** `{detail['signal_modifier']:.6f}`")
                st.write(f"**Disqualifier Penalty:** `-{detail['disqualifier_penalty']:.6f}`")
                
                # Show triggered penalties
                p_breakdown = json.loads(detail["penalties_breakdown"])
                if p_breakdown:
                    st.warning(f"**Triggered Disqualifiers:** {list(p_breakdown.keys())}")
                else:
                    st.success("**Triggered Disqualifiers:** None")
                    
            st.markdown("### Grounded Reasoning")
            st.info(f"\"{detail['reasoning']}\"")
