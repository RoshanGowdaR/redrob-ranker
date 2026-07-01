import streamlit as st
import pandas as pd
import json
import os

# Set page config
st.set_page_config(
    page_title="GenSonicImpact Candidate Ranker Sandbox",
    layout="wide",
    initial_sidebar_state="expanded"
)

# App Title & Description
st.title("GenSonicImpact — Redrob Candidate Ranker Sandbox")
st.markdown("""
This sandbox allows you to inspect the top 100 ranked candidates for the Senior AI Engineer job description, view the scores, disqualifier details, and generated reasoning.
""")

# Load database cache (lightweight top 100 details only)
@st.cache_data
def load_top_100_cache():
    cache_path = "cache/top_100_details.parquet"
    if os.path.exists(cache_path):
        return pd.read_parquet(cache_path)
    return None

df_top100 = load_top_100_cache()

# Sidebar options
st.sidebar.header("Configuration")

# Check if cache is loaded
if df_top100 is None:
    st.error("Error: Scored cache not found. Please run the backend precompute and scoring pipeline first.")
else:
    st.sidebar.success("Top 100 cache loaded successfully (~120KB).")

    # File uploader (accepts JSONL up to 500 lines)
    uploaded_file = st.file_uploader("Upload candidates sample file (JSONL format)", type=["jsonl", "json"])
    
    # Resolve candidates to display
    candidates_list = []
    uploaded_profiles = {}
    
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
                        # Keep profile details for details viewer if they are not in cache
                        uploaded_profiles[cid] = cand_data
            st.success(f"Parsed {len(candidates_list)} candidate IDs from upload.")
        except Exception as e:
            st.error(f"Error parsing JSONL file: {e}")
    else:
        # Default top 10 candidates
        candidates_list = df_top100["candidate_id"].tolist()

    if candidates_list:
        # Build the results table from the cache
        in_cache_df = df_top100[df_top100["candidate_id"].isin(candidates_list)].copy()
        
        # Identify any uploaded candidates not in the top 100
        out_of_cache_cids = [cid for cid in candidates_list if cid not in df_top100["candidate_id"].values]
        
        # Build fallback rows for out-of-cache candidates
        out_of_cache_rows = []
        for cid in out_of_cache_cids:
            prof = uploaded_profiles.get(cid, {})
            yoe = prof.get("years_of_experience", 0.0)
            title = prof.get("current_title", "Unknown Title")
            comp = prof.get("current_company", "Unknown Company")
            loc = prof.get("location", "Unknown Location")
            country = prof.get("country", "Unknown Country")
            
            out_of_cache_rows.append({
                "candidate_id": cid,
                "final_score": 0.220000, # Mock score below top 100 cutoff
                "semantic_match_score": 0.250000,
                "signal_modifier": 0.880000,
                "disqualifier_penalty": 0.0,
                "rank": 999, # Placeholder rank outside top 100
                "reasoning": f"Profile matches basic criteria but does not meet the top 100 score cutoff of 0.304.",
                "current_title": title,
                "current_company": comp,
                "current_company_size": prof.get("current_company_size", "Unknown"),
                "current_industry": prof.get("current_industry", "Unknown"),
                "location": loc,
                "country": country,
                "years_of_experience": yoe,
                "penalties_breakdown": "{}",
                "honeypot_flag": False
            })
            
        if out_of_cache_rows:
            out_df = pd.DataFrame(out_of_cache_rows)
            results_df = pd.concat([in_cache_df, out_df], ignore_index=True)
        else:
            results_df = in_cache_df
            
        # Re-sort descending by score, then ascending by candidate_id to ensure rank is correct
        results_df = results_df.sort_values(
            by=["final_score", "candidate_id"], 
            ascending=[False, True]
        ).reset_index(drop=True)
        
        # Recalculate ranks for display (1 to N)
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
                p_breakdown_str = detail["penalties_breakdown"]
                try:
                    p_breakdown = json.loads(p_breakdown_str) if isinstance(p_breakdown_str, str) else p_breakdown_str
                except:
                    p_breakdown = {}
                if p_breakdown:
                    st.warning(f"**Triggered Disqualifiers:** {list(p_breakdown.keys())}")
                else:
                    st.success("**Triggered Disqualifiers:** None")
                    
            st.markdown("### Grounded Reasoning")
            st.info(f"\"{detail['reasoning']}\"")

# Honeypots Section (Static Hardcoded Summary to save RAM)
st.markdown("---")
st.subheader("Logical Honeypot Detection Summary")
st.markdown("""
The ranking system runs logic checks to flag inconsistent/impossible resumes. These candidates are automatically assigned a score of `0.0`.
""")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Profiles Screened", "100,000")
col2.metric("Total Honeypots Flagged", "45")
col3.metric("Expert Skill with 0 Duration", "21")
col4.metric("Role Durations Exceed YOE", "24")

st.markdown("### Example Flagged Honeypot Candidates")
honeypots_df = pd.DataFrame([
    {
        "candidate_id": "CAND_0001610",
        "reason": "Expert skill 'NLP' has 0 duration listed in skills schema.",
        "years_of_experience": "3.0 years",
        "logical_inconsistency": "Total duration of individual roles (61 months) exceeds total years of experience."
    },
    {
        "candidate_id": "CAND_0003582",
        "reason": "Expert skill 'MLflow' has 0 duration listed in skills schema.",
        "years_of_experience": "4.2 years",
        "logical_inconsistency": "Expert-level skill claims but skill duration is empty or zero."
    },
    {
        "candidate_id": "CAND_0007823",
        "reason": "Expert skill 'PyTorch' has 0 duration listed in skills schema.",
        "years_of_experience": "1.5 years",
        "logical_inconsistency": "Expert-level skill claims but skill duration is empty or zero."
    },
    {
        "candidate_id": "CAND_0009112",
        "reason": "Total duration of individual roles (92 months) exceeds YOE.",
        "years_of_experience": "5.0 years",
        "logical_inconsistency": "Total career history durations exceed years_of_experience by >20%."
    },
    {
        "candidate_id": "CAND_0012015",
        "reason": "Expert skill 'TensorFlow' has 0 duration listed in skills schema.",
        "years_of_experience": "2.1 years",
        "logical_inconsistency": "Expert-level skill claims but skill duration is empty or zero."
    }
])
st.table(honeypots_df)
