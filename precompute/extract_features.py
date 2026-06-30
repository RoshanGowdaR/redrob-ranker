import os
import json
import pandas as pd
from tqdm import tqdm

def extract_features(candidates_file, out_file):
    print(f"Reading candidates from {candidates_file}...")
    records = []
    
    with open(candidates_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f, total=100000, desc="Extracting features"):
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            
            # Extract basic profile fields
            cid = data.get("candidate_id")
            profile = data.get("profile", {})
            yoe = profile.get("years_of_experience", 0.0)
            title = profile.get("current_title", "")
            company = profile.get("current_company", "")
            company_size = profile.get("current_company_size", "")
            industry = profile.get("current_industry", "")
            
            # 1. Career history text blob (all descriptions joined)
            career_history = data.get("career_history", [])
            descriptions = [entry.get("description", "").strip() for entry in career_history if entry.get("description")]
            career_history_text = " ".join(descriptions)
            
            # 2. Career history structured columns
            career_titles = [entry.get("title", "") for entry in career_history]
            career_companies = [entry.get("company", "") for entry in career_history]
            career_industries = [entry.get("industry", "") for entry in career_history]
            career_start_dates = [entry.get("start_date", "") for entry in career_history]
            career_end_dates = [entry.get("end_date") for entry in career_history]
            career_durations = [entry.get("duration_months", 0) for entry in career_history]
            career_is_current = [entry.get("is_current", False) for entry in career_history]
            
            # 3. Skills structured columns
            skills = data.get("skills", [])
            skill_names = [s.get("name", "") for s in skills]
            skill_proficiencies = [s.get("proficiency", "") for s in skills]
            skill_durations = [s.get("duration_months", 0) for s in skills]
            
            # 4. Redrob signals (23 fields)
            signals = data.get("redrob_signals", {})
            
            record = {
                "candidate_id": cid,
                "years_of_experience": yoe,
                "current_title": title,
                "current_company": company,
                "current_company_size": company_size,
                "current_industry": industry,
                
                # Career history
                "career_history_text": career_history_text,
                "career_history_json": json.dumps(career_history), # Full serialized structure
                "career_titles": career_titles,
                "career_companies": career_companies,
                "career_industries": career_industries,
                "career_start_dates": career_start_dates,
                "career_end_dates": career_end_dates,
                "career_durations": career_durations,
                "career_is_current": career_is_current,
                
                # Skills
                "skills_json": json.dumps(skills), # Full serialized structure
                "skill_names": skill_names,
                "skill_proficiencies": skill_proficiencies,
                "skill_durations": skill_durations,
                
                # 23 Redrob Signals
                "sig_profile_completeness_score": signals.get("profile_completeness_score", 0.0),
                "sig_signup_date": signals.get("signup_date", ""),
                "sig_last_active_date": signals.get("last_active_date", ""),
                "sig_open_to_work_flag": signals.get("open_to_work_flag", False),
                "sig_profile_views_received_30d": signals.get("profile_views_received_30d", 0),
                "sig_applications_submitted_30d": signals.get("applications_submitted_30d", 0),
                "sig_recruiter_response_rate": signals.get("recruiter_response_rate", 0.0),
                "sig_avg_response_time_hours": signals.get("avg_response_time_hours", 0.0),
                "sig_skill_assessment_scores": json.dumps(signals.get("skill_assessment_scores", {})),
                "sig_connection_count": signals.get("connection_count", 0),
                "sig_endorsements_received": signals.get("endorsements_received", 0),
                "sig_notice_period_days": signals.get("notice_period_days", 0),
                "sig_expected_salary_min": signals.get("expected_salary_range_inr_lpa", {}).get("min", 0.0),
                "sig_expected_salary_max": signals.get("expected_salary_range_inr_lpa", {}).get("max", 0.0),
                "sig_preferred_work_mode": signals.get("preferred_work_mode", ""),
                "sig_willing_to_relocate": signals.get("willing_to_relocate", False),
                "sig_github_activity_score": signals.get("github_activity_score", -1.0),
                "sig_search_appearance_30d": signals.get("search_appearance_30d", 0),
                "sig_saved_by_recruiters_30d": signals.get("saved_by_recruiters_30d", 0),
                "sig_interview_completion_rate": signals.get("interview_completion_rate", 0.0),
                "sig_offer_acceptance_rate": signals.get("offer_acceptance_rate", -1.0),
                "sig_verified_email": signals.get("verified_email", False),
                "sig_verified_phone": signals.get("verified_phone", False),
                "sig_linkedin_connected": signals.get("linkedin_connected", False)
            }
            records.append(record)
            
    df = pd.DataFrame(records)
    print(f"Saving {len(df)} rows to {out_file}...")
    df.to_parquet(out_file, index=False)
    print("Done!")
    
    # Print first 3 rows for verification
    print("\n--- FIRST 3 ROWS SAMPLE ---")
    pd.set_option('display.max_columns', None)
    print(df.head(3))

if __name__ == "__main__":
    os.makedirs("cache", exist_ok=True)
    extract_features("data/candidates.jsonl", "cache/features.parquet")
