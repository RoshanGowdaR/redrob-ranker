import os
import json
import numpy as np
import pandas as pd
from datetime import datetime

# Define individual penalty values as numbers
PENALTIES = {
    "pure_research": 0.30,        # Heavy
    "recent_langchain_only": 0.30, # Heavy
    "no_code_18m": 0.15,           # Moderate
    "title_hopping": 0.15,         # Moderate
    "framework_enthusiast": 0.05,  # Mild
    "consulting_only": 0.30,       # Heavy
    "wrong_domain": 0.15,          # Moderate
    "closed_source": 0.05,         # Mild
    "location_mismatch": 0.15      # Moderate
}

# Core vs Nice-to-have weights for 7 JD requirements
JD_WEIGHTS = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 0.3, 0.3])
JD_WEIGHTS_SUM = np.sum(JD_WEIGHTS)

def compute_semantic_scores(candidate_embeddings, req_embeddings):
    # Normalize candidate embeddings
    cand_norms = np.linalg.norm(candidate_embeddings, axis=1, keepdims=True)
    cand_norm = candidate_embeddings / (cand_norms + 1e-9)
    
    # Normalize requirement embeddings
    req_norms = np.linalg.norm(req_embeddings, axis=1, keepdims=True)
    req_norm = req_embeddings / (req_norms + 1e-9)
    
    # Cosine similarities (dot products since normalized)
    similarities = np.dot(cand_norm, req_norm.T) # shape (100000, 7)
    
    # Weighted average similarity
    weighted_similarities = np.dot(similarities, JD_WEIGHTS) / JD_WEIGHTS_SUM
    return weighted_similarities, similarities

def parse_date(date_str):
    if not date_str or pd.isna(date_str):
        return None
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        return None

def evaluate_disqualifiers(row):
    penalties_triggered = {}
    
    career = json.loads(row["career_history_json"])
    skills = json.loads(row["skills_json"])
    
    # Current date context (Today is June 30, 2026)
    today = datetime(2026, 6, 30)
    
    # 1. Pure-research-only career
    # All career entries read as academic/research with no production/deployment words
    is_research_only = len(career) > 0
    research_keywords = {"university", "college", "institute", "lab", "research", "academy", "phd", "scholar", "postdoc", "fellow", "academic"}
    prod_keywords = {"production", "deploy", "scale", "scalable", "shipped", "system", "infrastructure", "backend", "developer", "software", "product", "pipeline", "kubernetes", "docker", "cloud"}
    
    for role in career:
        comp = role.get("company", "").lower()
        title = role.get("title", "").lower()
        desc = role.get("description", "").lower()
        
        has_academic_indicator = any(kw in comp or kw in title for kw in research_keywords)
        has_prod_indicator = any(kw in desc or kw in title for kw in prod_keywords)
        
        # If it doesn't look academic, or it mentions production, then it's not research-only
        if not has_academic_indicator or has_prod_indicator:
            is_research_only = False
            break
            
    if is_research_only:
        penalties_triggered["pure_research"] = PENALTIES["pure_research"]
        
    # 2. Recent-LangChain-only AI experience
    # AI skills/descriptions only in roles starting in last 12 months (since June 30, 2025)
    # AND no pre-2022 ML/data production experience exists
    ai_keywords = {"langchain", "llm", "gpt", "openai", "generative ai", "rag", "prompt engineering", "fine-tuning", "llama", "claude"}
    ml_data_keywords = {"machine learning", "ml", "nlp", "computer vision", "data scientist", "data engineer", "data analyst", "python", "sql", "tensorflow", "pytorch", "scikit-learn", "pandas", "numpy"}
    
    has_ai_experience = False
    has_pre_july_2025_ai = False
    has_pre_2022_ml_data = False
    
    for role in career:
        start_dt = parse_date(role.get("start_date"))
        title = role.get("title", "").lower()
        desc = role.get("description", "").lower()
        
        role_has_ai = any(kw in desc or kw in title for kw in ai_keywords)
        role_has_ml_data = any(kw in desc or kw in title for kw in ml_data_keywords)
        
        if role_has_ai:
            has_ai_experience = True
            if start_dt and start_dt < datetime(2025, 7, 1):
                has_pre_july_2025_ai = True
                
        if role_has_ml_data and start_dt and start_dt < datetime(2022, 1, 1):
            has_pre_2022_ml_data = True
            
    # Check skills as well for AI names
    skill_names_lower = [s.lower() for s in row["skill_names"]]
    has_ai_skill = any(any(kw in sk for kw in ai_keywords) for sk in skill_names_lower)
    
    if has_ai_skill:
        has_ai_experience = True
        
    if has_ai_experience and not has_pre_july_2025_ai and not has_pre_2022_ml_data:
        penalties_triggered["recent_langchain_only"] = PENALTIES["recent_langchain_only"]
        
    # 3. No-code-in-18-months
    # Current title contains architect/lead/manager/head for 18+ months AND no coding language in description
    lead_titles = {"architect", "lead", "manager", "head of", "director", "vp", "chief"}
    code_langs = {"python", "java", "c++", "golang", "rust", "typescript", "javascript", "sql", "scala", "c#", "ruby"}
    
    no_code_triggered = False
    for role in career:
        if role.get("is_current") is True:
            title = role.get("title", "").lower()
            dur = role.get("duration_months", 0)
            desc = role.get("description", "").lower()
            
            is_lead = any(kw in title for kw in lead_titles)
            has_code_mention = any(kw in desc for kw in code_langs)
            
            if is_lead and dur >= 18 and not has_code_mention:
                no_code_triggered = True
                break
    if no_code_triggered:
        penalties_triggered["no_code_18m"] = PENALTIES["no_code_18m"]
        
    # 4. Title-hopping
    # 3+ companies in last 4-5 years, avg tenure < 18m, escalating Senior -> Staff -> Principal
    recent_roles = []
    for role in career:
        start_dt = parse_date(role.get("start_date"))
        if start_dt and start_dt >= datetime(2021, 6, 30):
            recent_roles.append(role)
            
    unique_companies = {r.get("company") for r in recent_roles if r.get("company")}
    if len(unique_companies) >= 3:
        tenures = [r.get("duration_months", 0) for r in recent_roles]
        avg_tenure = np.mean(tenures) if tenures else 0
        if avg_tenure < 18:
            # Check for title escalation keywords
            titles_sorted = sorted(recent_roles, key=lambda x: x.get("start_date", ""))
            titles_lower = [r.get("title", "").lower() for r in titles_sorted]
            
            # Simple check if there's escalation indicators
            has_sr = any("senior" in t or "sr" in t for t in titles_lower)
            has_staff = any("staff" in t for t in titles_lower)
            has_principal = any("principal" in t for t in titles_lower)
            
            # Escalation: either Sr then Staff, or Staff then Principal
            escalation = False
            for idx_a in range(len(titles_lower) - 1):
                t1 = titles_lower[idx_a]
                for idx_b in range(idx_a + 1, len(titles_lower)):
                    t2 = titles_lower[idx_b]
                    if ("senior" in t1 and "staff" in t2) or ("staff" in t1 and "principal" in t2) or ("senior" in t1 and "principal" in t2):
                        escalation = True
                        break
            
            if escalation:
                penalties_triggered["title_hopping"] = PENALTIES["title_hopping"]
                
    # 5. Framework-enthusiast
    # AI skills dominated by trendy framework names (duration <12m) AND no infra/systems skills
    trendy_frameworks = {"langchain", "llamaindex", "flowise", "autogpt", "huggingface", "gradio", "hugging face", "crewai"}
    systems_skills = {"pytorch", "tensorflow", "docker", "kubernetes", "spark", "kafka", "aws", "gcp", "azure", "pinecone", "faiss", "qdrant", "milvus", "weaviate", "elasticsearch", "opensearch", "redis", "postgres", "sql", "linux"}
    
    has_trendy = False
    has_systems = False
    
    for sk_name, sk_dur in zip(row["skill_names"], row["skill_durations"]):
        sk_lower = sk_name.lower()
        if any(f in sk_lower for f in trendy_frameworks):
            if sk_dur < 12:
                has_trendy = True
        if any(s in sk_lower for s in systems_skills):
            has_systems = True
            
    if has_trendy and not has_systems:
        penalties_triggered["framework_enthusiast"] = PENALTIES["framework_enthusiast"]
        
    # 6. Consulting-only career
    # Every single career company is in {TCS, Infosys, Wipro, Accenture, Cognizant, Capgemini}
    consulting_firms = {"tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini", "tata consultancy", "cognizant technology"}
    all_consulting = len(career) > 0
    for role in career:
        comp = role.get("company", "").lower()
        is_consulting = any(firm in comp for firm in consulting_firms)
        if not is_consulting:
            all_consulting = False
            break
    if all_consulting:
        penalties_triggered["consulting_only"] = PENALTIES["consulting_only"]
        
    # 7. Wrong-domain background
    # Dominant in CV, speech, or robotics with NO NLP/IR/text-retrieval language
    cv_speech_robotics = {"computer vision", "cv", "image classification", "object detection", "speech recognition", "tts", "stt", "robotics", "ros", "slam", "lidar", "opencv"}
    nlp_ir_text = {"nlp", "natural language processing", "information retrieval", "ir", "retrieval", "search", "rag", "embeddings", "vector search", "pinecone", "weaviate", "qdrant", "milvus", "faiss", "opensearch", "elasticsearch"}
    
    has_cv_speech_rob = False
    has_nlp_ir_text = False
    
    # Check skills
    for sk in row["skill_names"]:
        sk_lower = sk.lower()
        if any(kw in sk_lower for kw in cv_speech_robotics):
            has_cv_speech_rob = True
        if any(kw in sk_lower for kw in nlp_ir_text):
            has_nlp_ir_text = True
            
    # Check descriptions
    desc_all = row["career_history_text"].lower()
    if any(kw in desc_all for kw in nlp_ir_text):
        has_nlp_ir_text = True
        
    if has_cv_speech_rob and not has_nlp_ir_text:
        penalties_triggered["wrong_domain"] = PENALTIES["wrong_domain"]
        
    # 8. Closed-source-only
    # 5+ years experience, large company, github_activity = -1, and no certs/open-source mentions
    large_sizes = {"1001-5000", "5001-10000", "10001+"}
    is_large = row["current_company_size"] in large_sizes
    github_score = row["sig_github_activity_score"]
    
    if row["years_of_experience"] >= 5 and is_large and github_score == -1:
        # Check text descriptions for certifications or open source mentions
        text_to_check = (row["career_history_text"] + " " + row["current_title"]).lower()
        os_mentions = {"open source", "open-source", "github", "gitlab", "contributed", "contributor", "certification", "certified", "aws certified", "tensorFlow certified"}
        has_os_or_cert = any(kw in text_to_check for kw in os_mentions)
        
        if not has_os_or_cert:
            penalties_triggered["closed_source"] = PENALTIES["closed_source"]
            
    # 9. Location mismatch
    # Not in tier-1 Indian cities AND willing_to_relocate = false AND country != India
    tier1_cities = {"pune", "noida", "hyderabad", "mumbai", "delhi", "delhi ncr", "gurgaon", "bangalore", "bengaluru", "chennai", "kolkata"}
    loc_lower = row["current_title"].lower() # Wait, current_title doesn't have location! The features table doesn't have profile.location parsed as a column, but it has it in the JSON?
    # Wait, in extract_features.py, did we extract location?
    # Ah! In extract_features.py we extracted 'current_title', 'current_company', 'current_company_size', 'current_industry'. But wait, we did NOT parse location or country as separate columns!
    # Wait! They are in the candidate profile, but are they in the features table?
    # Let's check how we can get them.
    # Wait, we can parse location and country from the full career history or education JSON, or did the candidate schema have them in profile?
    # Yes! In the candidate schema, `profile` has `location` and `country`!
    # But in `extract_features.py` we did NOT save `location` and `country` as separate columns.
    # Ah! Let's check if we can read them from `career_history_json` or if we can reload the raw `candidates.jsonl` if needed, or if we can extract them.
    # Wait, we saved the full profile? No, we didn't save the full profile JSON, but wait! We can easily load location and country by writing a quick updater or since we are writing `scoring/score.py`, we can load the location and country from `data/candidates.jsonl` by candidate_id, or we can update `precompute/extract_features.py` to extract them and rebuild!
    # Wait, rebuilding features parquet takes only 13 seconds! Re-running Stage 1 is extremely fast and clean!
    # Yes, let's update `extract_features.py` to also save `location` and `country` as columns, and then run it again! That is the cleanest way.
    # Let's check if we can also get them from `sig_willing_to_relocate` and other fields.
    # Wait, does the feature table have `sig_willing_to_relocate`? Yes! It has `sig_willing_to_relocate` and `willing_to_relocate` flag.
    # Let's check: does the features table have country? No.
    # So updating `extract_features.py` to include `location` and `country` (from `profile`) is the best path.
    # Let's check where `profile` fields are. In `profile`: `location` and `country` are strings.
    # Let's write the code for `scoring/score.py` assuming they are in the parquet file, and we will update `extract_features.py` and run it again in 15 seconds!
    
    # We will assume columns "location" and "country" exist.
    loc = str(row.get("location", "")).lower()
    ctry = str(row.get("country", "")).lower()
    willing = row["sig_willing_to_relocate"]
    
    in_tier1 = any(city in loc for city in tier1_cities)
    is_india = (ctry == "india")
    
    # Location mismatch trigger:
    # Not in Tier-1 AND willing to relocate is False AND country is not India (meaning they are abroad, or inside India but not in Tier-1 and won't move)
    if not in_tier1 and not willing and not is_india:
        penalties_triggered["location_mismatch"] = PENALTIES["location_mismatch"]
        
    return penalties_triggered

def run_scoring(features_path, embeddings_path, req_embeddings_path, output_path=None):
    print("Loading features and embeddings...")
    df = pd.read_parquet(features_path)
    
    # Load candidate embeddings (check if split parts exist to fit within GitHub's 100MB limit)
    if embeddings_path.endswith("embeddings.npy"):
        part1_path = embeddings_path.replace(".npy", "_part1.npy")
        part2_path = embeddings_path.replace(".npy", "_part2.npy")
        if os.path.exists(part1_path) and os.path.exists(part2_path):
            candidate_embeddings = np.concatenate([np.load(part1_path), np.load(part2_path)], axis=0)
        else:
            candidate_embeddings = np.load(embeddings_path)
    else:
        candidate_embeddings = np.load(embeddings_path)
        
    req_embeddings = np.load(req_embeddings_path)
    
    # Compute semantic scores
    print("Computing semantic match scores...")
    semantic_scores, raw_sims = compute_semantic_scores(candidate_embeddings, req_embeddings)
    df["semantic_match_score"] = semantic_scores
    
    # Evaluate disqualifiers
    print("Evaluating disqualifiers for all candidates...")
    penalties_list = []
    penalties_breakdowns = []
    for idx, row in df.iterrows():
        p_dict = evaluate_disqualifiers(row)
        total_penalty = sum(p_dict.values())
        # Cap the total penalty at 0.6
        total_penalty = min(0.6, total_penalty)
        penalties_list.append(total_penalty)
        penalties_breakdowns.append(json.dumps(p_dict))
        
    df["disqualifier_penalty"] = penalties_list
    df["penalties_breakdown"] = penalties_breakdowns
    
    # Compute signal modifier
    print("Computing behavioral signal modifiers...")
    today = datetime(2026, 6, 30)
    
    modifiers = []
    for idx, row in df.iterrows():
        # Recency of last_active_date
        active_date = parse_date(row["sig_last_active_date"])
        if active_date:
            days_active = (today - active_date).days
            if days_active <= 30:
                sub_active = 1.0
            elif days_active <= 90:
                sub_active = 0.85
            elif days_active <= 180:
                sub_active = 0.70
            else:
                sub_active = 0.50
        else:
            sub_active = 0.50
            
        # Response rate
        resp_rate = row["sig_recruiter_response_rate"]
        if resp_rate >= 0.80:
            sub_response = 1.0
        elif resp_rate >= 0.50:
            sub_response = 0.85
        elif resp_rate >= 0.20:
            sub_response = 0.70
        else:
            sub_response = 0.50
            
        # Open to work
        sub_open = 1.0 if row["sig_open_to_work_flag"] else 0.85
        
        # Weighted sum modifier
        mod = 0.4 * sub_active + 0.4 * sub_response + 0.2 * sub_open
        modifiers.append(mod)
        
    df["signal_modifier"] = modifiers
    
    # Compute final score
    print("Computing final scores...")
    final_scores = df["semantic_match_score"] * df["signal_modifier"] - df["disqualifier_penalty"]
    
    # Force honeypot candidates to near-zero (exactly 0.0)
    final_scores = np.where(df["honeypot_flag"] == True, 0.0, final_scores)
    df["final_score"] = final_scores
    
    # Perform deterministic ranking (breaking ties by candidate_id ascending)
    print("Ranking candidates...")
    # Sort descending by final_score, then ascending by candidate_id
    df_sorted = df.sort_values(by=["final_score", "candidate_id"], ascending=[False, True]).reset_index(drop=True)
    df_sorted["rank"] = df_sorted.index + 1
    
    # Check for duplicate final_scores in the top 100
    top_100_scores = df_sorted.head(100)["final_score"].tolist()
    unique_scores_count = len(set(top_100_scores))
    print(f"\nVerification: In the top 100, there are {unique_scores_count} unique scores out of 100.")
    if unique_scores_count < 100:
        print("Ties exist, but they are resolved deterministically by candidate_id ascending.")
        
    # Print score distributions
    print("\n--- FINAL SCORE DISTRIBUTION ---")
    print(f"Min:  {df_sorted['final_score'].min():.4f}")
    print(f"Max:  {df_sorted['final_score'].max():.4f}")
    print(f"Mean: {df_sorted['final_score'].mean():.4f}")
    
    # Print top 20
    print("\n--- TOP 20 CANDIDATES ---")
    print(df_sorted[["rank", "candidate_id", "final_score", "semantic_match_score", "signal_modifier", "disqualifier_penalty", "honeypot_flag"]].head(20).to_string(index=False))
    
    # Print ranks 95 to 105 (boundary check)
    print("\n--- RANKS 95 TO 105 (CUTOFF BOUNDARY) ---")
    print(df_sorted[["rank", "candidate_id", "final_score", "semantic_match_score", "signal_modifier", "disqualifier_penalty", "honeypot_flag"]].iloc[94:105].to_string(index=False))
    
    if output_path:
        print(f"\nSaving scored features to {output_path}...")
        df_sorted.to_parquet(output_path, index=False)
        print("Done!")

if __name__ == "__main__":
    run_scoring(
        features_path="cache/features.parquet",
        embeddings_path="cache/embeddings.npy",
        req_embeddings_path="cache/requirement_embeddings.npy",
        output_path="cache/features_scored.parquet"
    )
