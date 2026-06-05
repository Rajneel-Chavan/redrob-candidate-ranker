# Redrob Hackathon — Intelligent Candidate Discovery & Ranking

**Author:** Rajneel Chavan | rajneelchavan16@gmail.com  
**Track:** Data & AI Challenge  
**Runtime:** ~33 seconds on CPU | No GPU | No external dependencies | No API calls

---

## Quick start

```bash
# Clone the repo
git clone https://github.com/rajneelchavan/redrob-candidate-ranker
cd redrob-candidate-ranker

# No pip install needed — stdlib only
python solution/rank.py --candidates ./candidates.jsonl --out ./rajneel_chavan.csv
```

That's it. One command, no setup.

---

## Reproduce the submission

```bash
python solution/rank.py --candidates ./candidates.jsonl --out ./rajneel_chavan.csv
```

Expected runtime: **~33 seconds** on a modern CPU (tested on Apple M2, 16 GB RAM).  
Output: `rajneel_chavan.csv` — 101 lines (header + 100 ranked candidates).

Validate output:
```bash
python validate_submission.py rajneel_chavan.csv
# → Submission is valid.
```

---

## Architecture

Multi-component feature scoring with a multiplicative behavioral modifier.  
Zero model loading. Zero external dependencies. Python stdlib only (`json`, `csv`, `datetime`, `pathlib`, `argparse`).

### Scoring formula

```
base_score = (
    0.22 × title_score          +   # Current role tier alignment
    0.25 × ai_experience_score  +   # AI/ML-specific years (not just total YoE)
    0.20 × skills_quality_score +   # Proficiency × endorsements × duration
    0.22 × career_content_score +   # Term scan of career descriptions
    0.08 × trajectory_score     +   # Product vs. consulting trajectory
    0.03 × location_score           # India-first per JD preference
)

final_score = base_score × behavioral_modifier
```

Top-100 scores are then linearly rescaled to `[0.10, 0.99]` to preserve relative order without ties.

### Title tiers

| Tier | Titles | Score |
|------|--------|-------|
| 1 | Search Engineer, Recommendation Systems Engineer, AI/ML Engineer, NLP Engineer, Machine Learning Engineer, Applied ML Engineer, Senior Applied Scientist | 1.00 |
| 2 | ML Engineer, AI Research Engineer, Data Scientist, AI Specialist, Computer Vision Engineer | 0.85 |
| 3 | Software Engineer, Backend Engineer, Analytics Engineer, Data Engineer (+ AI history check) | 0.40–0.65 |
| Non-technical | HR Manager, Business Analyst, Accountant, etc. | 0.05–0.20 |

### AI experience score

Doesn't just count total years — extracts AI/ML-specific years from `career_history`:
- Roles with AI/ML title keywords → 100% credit
- Roles with high-signal AI terms in description → 60% credit
- Roles in consulting firms → 0.65× penalty on recency weight

### Skills quality score

For each relevant skill: `weight × proficiency × (0.5 + 0.25×endorse_factor + 0.25×duration_factor)`

This catches keyword stuffers: a skill listed with `proficiency=beginner`, 0 endorsements, and 0 months used scores near zero regardless of the skill name.

### Career content score

Weighted term scan across all career role descriptions. High-signal terms (NDCG, MRR, A/B testing, FAISS, Qdrant, production, shipped) are weighted 3.5–5.0. Common terms (model, training) are 1.0. Recency-weighted: most recent role counts 3× more than a role from 5 jobs ago.

### Behavioral modifier

Multiplicative factor `[0.25, 1.30]` combining 7 signals:
- `open_to_work_flag` — ×1.12 or ×0.88
- Days since `last_active_date` — ×1.18 (≤14d) down to ×0.62 (>180d)
- `recruiter_response_rate` — ×1.12 (≥75%) down to ×0.75 (<15%)
- `notice_period_days` — ×1.12 (≤15d) down to ×0.85 (>90d)
- `interview_completion_rate` — ×1.06 (≥85%) or ×0.92 (<30%)
- `github_activity_score` — ×1.06 (≥70)
- `saved_by_recruiters_30d` — ×1.04 (≥5)

### Honeypot detection

Fires on any of:
1. Total career months > 2.2× claimed `years_of_experience`
2. `expert` proficiency on a skill with `duration_months = 0` (×3 or more)
3. 8+ skills at `expert` level
4. Total skill endorsements > 5× connection count
5. Multiple `is_current = true` roles simultaneously

Honeypots receive score 0.001 and never appear in the top 100.

---

## Results

- **Runtime:** 33 seconds on Apple M2 CPU (well within 5-minute constraint)
- **Top 100 titles:** 100% Tier-1/2 AI/ML titles (Search Engineer, Recommendation Systems Engineer, Machine Learning Engineer, AI Engineer, Applied ML Engineer, NLP Engineer, Data Scientist)
- **India-based:** 87/100 candidates
- **Honeypots:** 0/100

---

## Files

```
solution/
  rank.py                     # Main ranking script (the only file you need)
  requirements.txt            # Empty — no external dependencies
  submission_metadata.yaml    # Hackathon submission metadata
  README.md                   # This file
  rajneel_chavan.csv          # Submission output (100 ranked candidates)
```

---

## Why this approach

The JD explicitly warns: *"The right answer is not 'find candidates whose skills section contains the most AI keywords.'"*

Key design choices made to respect this:
1. **Title is the anchor signal** — a Marketing Manager with 10 AI skills is not a Senior AI Engineer candidate. Title tier is the most decisive component.
2. **Career descriptions > skill lists** — terms found in career history descriptions (NDCG, production, shipped, FAISS) are much harder to fake than skill list keywords.
3. **Behavioral signals are a multiplier, not additive** — a perfect-on-paper candidate who hasn't logged in for 7 months and responds to 5% of recruiter messages is, for practical hiring, unavailable. Their base score is multiplied down, not just adjusted.
4. **Consulting-only penalty** — the JD explicitly disqualifies candidates whose entire career is at TCS/Infosys/Wipro/Accenture/etc. Career descriptions from consulting roles receive a 0.65× recency weight.
5. **No LLM calls at inference time** — aligns with the production constraint and demonstrates that good ML engineering doesn't always need a language model.
