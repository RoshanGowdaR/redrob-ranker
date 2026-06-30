import os
import json
import pandas as pd
import numpy as np

def run_honeypot_checks(parquet_path):
    print(f"Loading features from {parquet_path}...")
    df = pd.read_parquet(parquet_path)
    
    honeypot_flags = []
    honeypot_reasons = []
    
    # Track statistics
    stats = {
        "rule_a": 0,
        "rule_b": 0,
        "rule_c": 0,
        "rule_d": 0,
        "total_flagged": 0
    }
    
    for idx, row in df.iterrows():
        reasons = []
        
        # Parse JSON structures
        skills = json.loads(row["skills_json"])
        career = json.loads(row["career_history_json"])
        yoe = row["years_of_experience"]
        
        # Rule (a): Skill has proficiency "expert" but duration_months is 0 or null
        rule_a_triggered = False
        for s in skills:
            prof = s.get("proficiency", "").lower()
            dur = s.get("duration_months")
            if prof == "expert" and (dur is None or dur == 0):
                rule_a_triggered = True
                reasons.append(f"Expert skill '{s.get('name')}' has 0/null duration")
                break
        if rule_a_triggered:
            stats["rule_a"] += 1
            
        # Rule (b): Sum of career_history duration_months exceeds years_of_experience * 12 by more than 20%
        durations = [c.get("duration_months", 0) for c in career]
        total_career_months = sum(durations)
        allowed_months = yoe * 12 * 1.20
        if total_career_months > allowed_months:
            reasons.append(f"Total career history duration ({total_career_months}m) exceeds years of experience ({yoe}y, limit {allowed_months:.1f}m)")
            stats["rule_b"] += 1
            
        # Rule (c): Single career history entry duration exceeds years of experience
        # (This replaces the founding-date check which is not present in the schema)
        max_single_month = max(durations) if durations else 0
        # Allow 1 month buffer for rounding errors
        if max_single_month > (yoe * 12) + 1:
            reasons.append(f"Longest single role duration ({max_single_month}m) exceeds total years of experience ({yoe}y)")
            stats["rule_c"] += 1
            
        # Rule (d): Overlapping is_current roles across different companies
        current_roles = [c for c in career if c.get("is_current") is True]
        current_companies = {c.get("company") for c in current_roles if c.get("company")}
        if len(current_companies) > 1:
            reasons.append(f"Multiple active roles (is_current=True) at different companies: {list(current_companies)}")
            stats["rule_d"] += 1
            
        if reasons:
            honeypot_flags.append(True)
            honeypot_reasons.append("; ".join(reasons))
            stats["total_flagged"] += 1
        else:
            honeypot_flags.append(False)
            honeypot_reasons.append("")
            
    df["honeypot_flag"] = honeypot_flags
    df["honeypot_reason"] = honeypot_reasons
    
    print("\n--- HONEYPOT CHECK STATISTICS ---")
    print(f"Total candidates: {len(df)}")
    print(f"Rule (a) flags (expert with 0 duration): {stats['rule_a']}")
    print(f"Rule (b) flags (total duration > YOE * 12 * 1.2): {stats['rule_b']}")
    print(f"Rule (c) flags (single role duration > YOE * 12): {stats['rule_c']}")
    print(f"Rule (d) flags (multiple current roles): {stats['rule_d']}")
    print(f"Total unique candidates flagged: {stats['total_flagged']} ({stats['total_flagged'] / len(df) * 100:.3f}%)")
    
    # Save back to cache
    print(f"\nSaving updated features with honeypot flags to {parquet_path}...")
    df.to_parquet(parquet_path, index=False)
    print("Done!")

if __name__ == "__main__":
    run_honeypot_checks("cache/features.parquet")
