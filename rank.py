#!/usr/bin/env python3
"""
Redrob Hackathon — Intelligent Candidate Discovery & Ranking
Author: Rajneel Chavan | rajneelchavan16@gmail.com

Architecture: Multi-component feature scoring + behavioral signal modifier.
No external API calls. No GPU. Runs CPU-only in < 5 minutes on 100K candidates.

Scoring formula:
  base_score = 0.22*title + 0.25*ai_exp + 0.20*skills_quality + 0.22*career_content
              + 0.08*trajectory + 0.03*location
  final_score = base_score × behavioral_modifier

Honeypots (impossible profiles) are force-ranked to 0.001.
"""

import json
import csv
import sys
import re
import argparse
from datetime import date
from pathlib import Path
from typing import Dict, Any, List, Tuple

# ============================================================================
# JD-Derived Constants
# ============================================================================

TODAY = date.today()

# Consulting firms explicitly flagged in JD as red flags for career-only candidates
CONSULTING_FIRMS = frozenset({
    'tcs', 'tata consultancy', 'infosys', 'wipro', 'accenture', 'cognizant',
    'capgemini', 'hcl technologies', 'hcl tech', 'tech mahindra', 'mphasis',
    'hexaware', 'ltimindtree', 'l&t infotech', 'niit technologies', 'zensar',
    'mastech', 'igate', 'syntel', 'patni', 'firstsource', 'birlasoft',
    'mindtree',   # now LTIMindtree
})

# --- Title tiers (derived from full dataset title list) ---

TIER1_TITLES = frozenset({
    # Direct JD match — ranking / search / recommendation at product companies
    'search engineer', 'recommendation systems engineer',
    'ai engineer', 'senior ai engineer', 'lead ai engineer', 'staff ai engineer',
    'nlp engineer', 'senior nlp engineer',
    'machine learning engineer', 'senior machine learning engineer',
    'staff machine learning engineer', 'applied ml engineer',
    'senior applied scientist', 'applied scientist',
    'ai/ml engineer', 'ml infrastructure engineer',
    'relevance engineer', 'ranking engineer', 'personalization engineer',
})

TIER2_TITLES = frozenset({
    # Strong signal — production ML but slightly less aligned
    'ml engineer', 'ai research engineer', 'data scientist',
    'senior data scientist', 'ai specialist',
    'computer vision engineer',          # CV background; need IR pivot shown in desc
    'senior software engineer (ml)',     # Hybrid title; desc decides
    'research scientist', 'research engineer',
    'deep learning engineer',
})

TIER3_TITLES = frozenset({
    # Adjacent — need strong career description to compensate
    'junior ml engineer',                # Too junior; only if YoE mismatch
    'analytics engineer', 'data engineer', 'data analyst', 'senior data engineer',
    'backend engineer', 'software engineer', 'senior software engineer',
    'full stack developer', 'cloud engineer',
    'java developer', '.net developer', 'devops engineer',
    'mobile developer', 'frontend engineer', 'qa engineer',
})

# Non-technical titles — almost certain mismatches per JD
NON_TECHNICAL_TITLES = frozenset({
    'business analyst', 'hr manager', 'mechanical engineer', 'accountant',
    'project manager', 'customer support', 'operations manager', 'content writer',
    'sales executive', 'civil engineer', 'graphic designer', 'marketing manager',
})

# --- Skills relevant to this JD, with importance weights ---
# Key: normalized skill name (lowercase), Value: weight 1.0–3.5
RELEVANT_SKILLS: Dict[str, float] = {
    # Retrieval & Search (highest — JD's core requirement)
    'information retrieval': 3.5, 'vector search': 3.5, 'semantic search': 3.5,
    'hybrid search': 3.5, 'bm25': 3.0, 'reranking': 3.5, 'cross-encoder': 3.5,
    'bi-encoder': 3.0, 'dense retrieval': 3.0, 'sparse retrieval': 3.0,
    'faiss': 3.5, 'pinecone': 3.5, 'qdrant': 3.5, 'weaviate': 3.5,
    'milvus': 3.5, 'elasticsearch': 3.0, 'opensearch': 3.0, 'pgvector': 2.5,

    # Ranking / Recommendation (core JD role)
    'ranking': 3.5, 'learning to rank': 3.5, 'ltr': 3.5,
    'recommendation systems': 3.0, 'collaborative filtering': 2.5,
    'personalization': 2.5,

    # Embeddings & Representation Learning
    'embeddings': 3.5, 'sentence transformers': 3.5, 'text embeddings': 3.0,
    'hugging face transformers': 3.0, 'fine-tuning llms': 3.0,
    'llms': 3.0, 'rag': 3.0, 'peft': 2.5,

    # NLP (JD specifically mentions NLP/IR alignment)
    'nlp': 3.0, 'natural language processing': 3.0,
    'bert': 2.5, 'transformers': 2.5,

    # Evaluation — explicitly called out in JD ("NDCG, MRR, MAP")
    'ndcg': 4.0, 'mrr': 4.0, 'map': 3.5, 'a/b testing': 3.5,
    'offline evaluation': 3.5, 'online evaluation': 3.5,

    # ML Frameworks
    'pytorch': 2.5, 'tensorflow': 2.0, 'scikit-learn': 2.0, 'xgboost': 2.0,
    'lightgbm': 2.0, 'mlflow': 2.0, 'kubeflow': 1.5, 'weights & biases': 1.5,

    # MLOps / Deployment (production signal)
    'model deployment': 2.5, 'model serving': 2.5, 'docker': 1.5,
    'kubernetes': 1.5, 'aws': 1.5, 'gcp': 1.5,

    # Core
    'python': 2.0, 'machine learning': 2.0, 'deep learning': 2.0,
    'langchain': 1.5, 'langgraph': 2.0, 'haystack': 1.5,

    # CV skills (adjacent, lower weight for this NLP/IR role)
    'computer vision': 1.0, 'object detection': 0.8, 'image classification': 0.8,
}

# Terms to find in career history descriptions, with importance weights
CAREER_TERMS: Dict[str, float] = {
    # Retrieval/Ranking/Recommendation — the heart of this role
    'retrieval': 3.5, 'ranking': 3.5, 'recommendation': 3.5, 'rerank': 3.5,
    'embedding': 3.5, 'embeddings': 3.5, 'vector': 3.0,
    'semantic search': 4.0, 'information retrieval': 4.0,
    'hybrid search': 4.0, 'bm25': 3.5,
    'cross-encoder': 3.5, 'bi-encoder': 3.0,
    'faiss': 4.0, 'pinecone': 4.0, 'qdrant': 4.0, 'weaviate': 4.0,
    'milvus': 4.0, 'elasticsearch': 3.0, 'opensearch': 3.0,
    'learning to rank': 4.0, 'xgboost': 2.0, 'lightgbm': 2.0,

    # Evaluation — JD calls these out by name
    'ndcg': 5.0, 'mrr': 5.0, 'map': 4.0,
    'a/b test': 4.5, 'a/b testing': 4.5,
    'offline eval': 4.0, 'online eval': 4.0, 'evaluation framework': 4.0,
    'recall@': 4.0, 'precision@': 4.0, 'offline': 2.5, 'online': 2.5,

    # Production signals (JD heavily weights this)
    'production': 3.0, 'deployed': 3.0, 'shipped': 3.0,
    'real users': 3.5, 'at scale': 3.0, 'serving': 2.5,
    'latency': 2.5, 'inference': 2.5, 'throughput': 2.0,
    'millions': 2.5, 'billion': 2.5, '10m+': 3.0, '100k': 2.5,

    # LLM/NLP
    'llm': 3.0, 'language model': 3.0, 'rag': 3.5,
    'fine-tuning': 3.0, 'fine-tune': 3.0, 'lora': 2.5, 'qlora': 2.5, 'rlhf': 3.0,
    'transformer': 2.5, 'bert': 2.5, 'nlp': 2.5, 'natural language': 2.5,
    'sentence-transformer': 3.5, 'sentence transformer': 3.5,

    # Product company signals (vs consulting)
    'startup': 2.0, 'series a': 2.5, 'series b': 2.0, 'product': 1.5,
    'launched': 2.0, 'revenue': 1.5, 'growth': 1.5, 'user': 1.5,

    # Base ML (lower weight — very common, less differentiating)
    'machine learning': 1.5, 'deep learning': 1.5, 'neural': 1.5,
    'model': 1.0, 'training': 1.0, 'predict': 1.0, 'feature': 1.0,
}

# India locations preferred by the JD
INDIA_CITIES = frozenset({
    'pune', 'noida', 'bangalore', 'bengaluru', 'hyderabad', 'mumbai',
    'delhi', 'gurgaon', 'gurugram', 'chennai', 'kolkata', 'india',
    'trivandrum', 'kochi', 'ahmedabad', 'jaipur', 'chandigarh',
    'indore', 'bhopal', 'nagpur', 'coimbatore', 'madurai', 'vizag',
})

PROFICIENCY_MAP = {'beginner': 0.25, 'intermediate': 0.50, 'advanced': 0.80, 'expert': 1.00}


# ============================================================================
# Scoring Components
# ============================================================================

def get_title_score(current_title: str, career_history: List[Dict]) -> float:
    """
    Score based on current role alignment with the Senior AI Engineer JD.
    Past career titles boost Software Engineers who have pivoted into AI/ML.
    """
    ct = current_title.lower().strip()

    if ct in TIER1_TITLES:
        return 1.00

    if ct in TIER2_TITLES:
        return 0.85

    # Tier 3: technical but requires strong AI evidence in career history
    if ct in TIER3_TITLES:
        # Check if past roles show AI/ML progression
        ai_title_keywords = {'ml', 'ai', 'machine learning', 'data scientist', 'nlp',
                             'deep learning', 'research', 'ranking', 'search', 'recommendation'}
        has_ai_history = any(
            any(kw in r.get('title', '').lower() for kw in ai_title_keywords)
            for r in career_history
        )
        return 0.65 if has_ai_history else 0.40

    # Non-technical: check if they have any AI/ML career history (career pivot)
    if ct in NON_TECHNICAL_TITLES:
        has_ai_history = any(
            any(kw in r.get('title', '').lower()
                for kw in {'ml', 'ai', 'machine learning', 'data scientist', 'engineer'})
            for r in career_history
        )
        return 0.20 if has_ai_history else 0.05

    # Unknown title — partial credit
    return 0.40


def get_experience_score(yoe: float) -> float:
    """Score total years of experience. JD sweet spot: 5–9 years."""
    if 5.0 <= yoe <= 9.0:
        return 1.00
    elif 4.0 <= yoe < 5.0:
        return 0.88
    elif 9.0 < yoe <= 12.0:
        return 0.88
    elif 3.0 <= yoe < 4.0:
        return 0.72
    elif 12.0 < yoe <= 15.0:
        return 0.72
    elif 2.0 <= yoe < 3.0:
        return 0.50
    elif yoe > 15.0:
        return 0.62
    else:
        return 0.28


def get_ai_experience_score(career_history: List[Dict]) -> float:
    """
    Estimate years of specifically AI/ML-relevant experience from career history.
    This separates candidates who have been in AI for 5 years from those who
    recently added AI keywords after a non-AI career.
    """
    AI_TITLE_KWS = frozenset({
        'ml', 'ai', 'machine learning', 'data scientist', 'nlp', 'deep learning',
        'research scientist', 'applied scientist', 'ranking', 'search',
        'recommendation', 'retrieval', 'embedding',
    })
    ai_months = 0.0

    for role in career_history:
        title_lower = role.get('title', '').lower()
        desc_lower = role.get('description', '').lower()
        duration = max(0, role.get('duration_months', 0))

        is_ai_title = any(kw in title_lower for kw in AI_TITLE_KWS)

        # Count high-signal AI terms in description
        ai_term_hits = sum(
            1 for term, w in CAREER_TERMS.items()
            if w >= 2.5 and term in desc_lower
        )
        is_ai_content = ai_term_hits >= 3

        if is_ai_title:
            ai_months += duration
        elif is_ai_content:
            ai_months += duration * 0.60   # Partial credit for AI-adjacent content

    ai_years = ai_months / 12.0

    if ai_years >= 5.0:
        return 1.00
    elif ai_years >= 4.0:
        return 0.90
    elif ai_years >= 3.0:
        return 0.78
    elif ai_years >= 2.0:
        return 0.60
    elif ai_years >= 1.0:
        return 0.40
    elif ai_years >= 0.5:
        return 0.25
    else:
        return 0.10


def get_skills_quality_score(skills: List[Dict]) -> Tuple[float, List[str]]:
    """
    Score skills on three axes:
      1. Relevance to this JD (weight from RELEVANT_SKILLS)
      2. Proficiency level (self-reported but still signal)
      3. Endorsements + usage duration (trust multiplier)

    Returns (normalized_score, top_skill_names_list)
    """
    skill_scores = []

    for skill in skills:
        name = skill.get('name', '').lower().strip()
        relevance = 0.0
        for rel_name, w in RELEVANT_SKILLS.items():
            if rel_name in name or name in rel_name:
                relevance = max(relevance, w)

        if relevance == 0:
            continue

        proficiency = PROFICIENCY_MAP.get(skill.get('proficiency', 'beginner'), 0.25)

        endorsements = skill.get('endorsements', 0)
        endorse_factor = min(1.0, (endorsements ** 0.5) / 6.0)   # sqrt(36)=6 → 1.0

        duration_months = skill.get('duration_months', 0)
        duration_factor = min(1.0, duration_months / 24.0)        # 2+ years → 1.0

        # Combined: relevance × proficiency × trust
        score = relevance * proficiency * (0.50 + 0.25 * endorse_factor + 0.25 * duration_factor)
        skill_scores.append((score, skill.get('name', name)))

    if not skill_scores:
        return 0.0, []

    skill_scores.sort(reverse=True)
    top_names = [n for _, n in skill_scores[:5]]

    # Normalize: perfect score = top-5 all at max weight(3.5) × 1.0 × 1.0 = 3.5 each
    MAX = 3.5 * 5
    total = sum(s for s, _ in skill_scores[:5])
    return min(1.0, total / MAX), top_names


def get_career_content_score(career_history: List[Dict]) -> Tuple[float, List[str]]:
    """
    Score the AI/ML content density in career role descriptions.
    Recency-weighted: current/recent roles count more.
    Consulting-company roles get a penalty since the JD disqualifies consulting-only careers.

    Returns (score, list_of_top_matched_terms)
    """
    if not career_history:
        return 0.0, []

    sorted_roles = sorted(
        career_history,
        key=lambda r: r.get('start_date', '2000-01-01'),
        reverse=True
    )

    total_weighted = 0.0
    total_weight = 0.0
    matched_terms: Dict[str, float] = {}

    for i, role in enumerate(sorted_roles):
        recency_w = max(0.30, 1.0 - i * 0.15)

        # Consulting penalty
        company = role.get('company', '').lower()
        if any(firm in company for firm in CONSULTING_FIRMS):
            recency_w *= 0.65

        desc = role.get('description', '').lower()
        role_score = 0.0
        for term, w in CAREER_TERMS.items():
            if term in desc:
                role_score += w
                if w >= 3.0:
                    matched_terms[term] = max(matched_terms.get(term, 0), w)

        # Normalize per-role (cap denominator based on typical max achievable)
        role_score_norm = min(1.0, role_score / 30.0)

        total_weighted += role_score_norm * recency_w
        total_weight += recency_w

    if total_weight == 0:
        return 0.0, []

    final = min(1.0, total_weighted / total_weight)
    top_terms = [t for t, _ in sorted(matched_terms.items(), key=lambda x: -x[1])[:4]]
    return final, top_terms


def get_trajectory_score(career_history: List[Dict]) -> float:
    """
    Penalize all-consulting careers (JD explicitly disqualifies).
    Reward product-company AI/ML progression.
    """
    if not career_history:
        return 0.3

    total_months = sum(max(0, r.get('duration_months', 0)) for r in career_history)
    if total_months == 0:
        return 0.3

    consulting_months = 0
    product_ai_months = 0

    AI_TITLE_KWS = frozenset({'ml', 'ai', 'machine learning', 'data scientist', 'nlp',
                               'research', 'ranking', 'search', 'recommendation'})

    for role in career_history:
        company = role.get('company', '').lower()
        title = role.get('title', '').lower()
        duration = max(0, role.get('duration_months', 0))

        is_consulting = any(firm in company for firm in CONSULTING_FIRMS)
        is_ai_title = any(kw in title for kw in AI_TITLE_KWS)

        if is_consulting:
            consulting_months += duration
        if is_ai_title and not is_consulting:
            product_ai_months += duration

    consulting_ratio = consulting_months / total_months
    product_ai_ratio = product_ai_months / total_months

    # Heavy penalty for all-consulting background
    if consulting_ratio >= 0.90:
        return 0.15
    elif consulting_ratio >= 0.70:
        return 0.35
    elif consulting_ratio >= 0.50:
        return 0.55

    # Reward product AI experience
    return min(1.0, 0.55 + product_ai_ratio * 0.45)


def get_location_score(profile: Dict, signals: Dict) -> float:
    """India-based preferred (JD: Pune/Noida, open to Bangalore/Mumbai/Delhi/Hyderabad)."""
    location = profile.get('location', '').lower()
    country = profile.get('country', '').lower()

    if country == 'india' or any(city in location for city in INDIA_CITIES):
        # Bonus for JD's preferred cities
        preferred = {'pune', 'noida', 'bangalore', 'bengaluru', 'delhi', 'gurgaon'}
        if any(c in location for c in preferred):
            return 1.0
        return 0.95

    if signals.get('willing_to_relocate', False):
        return 0.70

    return 0.45


def get_behavioral_modifier(signals: Dict) -> float:
    """
    Multiplicative behavioral modifier (range ~0.40–1.45).

    Combines 7 behavioral signals that indicate candidate availability and
    engagement quality — exactly the kind of signal that separates a perfect-
    on-paper candidate from one who will actually interview and accept.
    """
    m = 1.0

    # 1. Open to work (direct availability signal)
    if signals.get('open_to_work_flag', False):
        m *= 1.12
    else:
        m *= 0.88

    # 2. Recency of last activity
    last_active_str = signals.get('last_active_date', '')
    if last_active_str:
        try:
            last_active = date.fromisoformat(last_active_str)
            days_ago = (TODAY - last_active).days
            if days_ago <= 14:
                m *= 1.18
            elif days_ago <= 30:
                m *= 1.10
            elif days_ago <= 60:
                m *= 1.02
            elif days_ago <= 90:
                m *= 0.95
            elif days_ago <= 180:
                m *= 0.82
            else:
                m *= 0.62   # > 6 months inactive = major availability concern
        except ValueError:
            pass

    # 3. Recruiter response rate (will they reply when we reach out?)
    rr = signals.get('recruiter_response_rate', 0.5)
    if rr >= 0.75:
        m *= 1.12
    elif rr >= 0.50:
        m *= 1.05
    elif rr >= 0.30:
        m *= 0.98
    elif rr >= 0.15:
        m *= 0.88
    else:
        m *= 0.75

    # 4. Notice period (how quickly can they start?)
    notice = signals.get('notice_period_days', 60)
    if notice <= 15:
        m *= 1.12
    elif notice <= 30:
        m *= 1.07
    elif notice <= 60:
        m *= 1.00
    elif notice <= 90:
        m *= 0.94
    else:
        m *= 0.85

    # 5. Interview completion rate (reliability of showing up)
    icr = signals.get('interview_completion_rate', 0.5)
    if icr >= 0.85:
        m *= 1.06
    elif icr < 0.30:
        m *= 0.92

    # 6. GitHub activity (technical engagement signal, JD values open-source)
    github_score = signals.get('github_activity_score', -1)
    if github_score >= 70:
        m *= 1.06
    elif github_score >= 40:
        m *= 1.02
    elif github_score == -1:
        pass  # No GitHub linked — neutral
    # Low GitHub (< 10) is not strongly penalized; not everyone OSS

    # 7. Saved by recruiters in last 30 days (social proof)
    saved = signals.get('saved_by_recruiters_30d', 0)
    if saved >= 5:
        m *= 1.04

    return max(0.25, min(1.30, m))


def get_assessment_bonus(signals: Dict) -> float:
    """
    Small bonus for platform-verified skill assessment scores.
    Only count assessments for skills relevant to this JD.
    """
    assessments = signals.get('skill_assessment_scores', {})
    if not assessments:
        return 0.0

    RELEVANT_ASSESSMENT_TERMS = frozenset({
        'faiss', 'qdrant', 'pinecone', 'weaviate', 'elasticsearch', 'opensearch',
        'embeddings', 'sentence transformers', 'information retrieval',
        'nlp', 'machine learning', 'deep learning', 'pytorch', 'tensorflow',
        'llm', 'rag', 'fine-tuning', 'peft', 'transformer', 'bert',
        'recommendation', 'ranking', 'vector search', 'semantic search',
        'mlflow', 'kubeflow', 'weights & biases',
    })

    relevant_scores = []
    for skill_name, score in assessments.items():
        skill_lower = skill_name.lower()
        if any(rt in skill_lower for rt in RELEVANT_ASSESSMENT_TERMS):
            relevant_scores.append(score / 100.0)

    if not relevant_scores:
        return 0.0
    return sum(relevant_scores) / len(relevant_scores) * 0.05   # cap at +5% bonus


# ============================================================================
# Honeypot Detection
# ============================================================================

def is_honeypot(candidate: Dict) -> bool:
    """
    Detect profiles with impossible or internally inconsistent data.
    The dataset contains ~80 honeypots; submissions with >10% in top-100
    are auto-disqualified.

    Checks:
      1. Career timeline total >> claimed years_of_experience
      2. Expert proficiency on a skill with 0 months usage (impossible)
      3. 8+ expert-level skills (statistically implausible)
      4. More skill endorsements received than connection count allows
      5. Multiple 'is_current' roles simultaneously (can't hold 2+ current jobs)
    """
    profile = candidate['profile']
    career_history = candidate.get('career_history', [])
    skills = candidate.get('skills', [])
    signals = candidate.get('redrob_signals', {})

    # Check 1: Timeline impossibility
    total_career_months = sum(max(0, r.get('duration_months', 0)) for r in career_history)
    claimed_months = profile.get('years_of_experience', 0) * 12
    if claimed_months > 6 and total_career_months > claimed_months * 2.2:
        return True

    # Check 2: Expert with 0 months used
    expert_zero = sum(
        1 for s in skills
        if s.get('proficiency') == 'expert' and s.get('duration_months', 0) == 0
    )
    if expert_zero >= 3:
        return True

    # Check 3: Too many expert skills
    expert_count = sum(1 for s in skills if s.get('proficiency') == 'expert')
    if expert_count >= 8:
        return True

    # Check 4: Endorsements impossible given connections
    connections = signals.get('connection_count', 0)
    total_endorsements = sum(s.get('endorsements', 0) for s in skills)
    if connections > 0 and total_endorsements > connections * 5:
        return True
    if connections < 30 and total_endorsements > 300:
        return True

    # Check 5: Multiple simultaneous current roles
    current_roles = sum(1 for r in career_history if r.get('is_current', False))
    if current_roles > 1:
        return True

    return False


# ============================================================================
# Main Scoring
# ============================================================================

def score_candidate(candidate: Dict) -> Tuple[float, Dict]:
    """
    Compute composite score for one candidate.
    Returns (final_score, breakdown_dict).

    Fast path: non-technical titles with no AI skills get a cheap score
    (~0.02–0.12) without running the expensive career-content scan.
    This cuts total runtime by ~3x since 65K of 100K candidates are non-technical.
    """
    profile = candidate['profile']
    current_title = profile.get('current_title', '').lower().strip()
    signals = candidate.get('redrob_signals', {})

    # Fast path: clearly non-technical candidates
    if current_title in NON_TECHNICAL_TITLES:
        skills = candidate.get('skills', [])
        skill_names = {s.get('name', '').lower() for s in skills}
        AI_SKILL_SET = frozenset({'information retrieval', 'semantic search',
                                   'embeddings', 'llms', 'faiss', 'rag',
                                   'vector search', 'recommendation systems',
                                   'sentence transformers', 'fine-tuning llms'})
        if not (skill_names & AI_SKILL_SET):
            # No AI skills at all → tiny score, no full scan needed
            behavioral = get_behavioral_modifier(signals)
            return 0.05 * behavioral, {'honeypot': False, 'fast_path': True,
                                        'top_skills': [], 'career_terms': []}

    career = candidate.get('career_history', [])
    skills = candidate.get('skills', [])
    education = candidate.get('education', [])

    # Short-circuit on honeypots
    if is_honeypot(candidate):
        return 0.0010, {'honeypot': True}

    # Component scores
    title_score = get_title_score(profile.get('current_title', ''), career)
    exp_score = get_experience_score(profile.get('years_of_experience', 0.0))
    ai_exp_score = get_ai_experience_score(career)
    skills_score, top_skills = get_skills_quality_score(skills)
    career_score, career_terms = get_career_content_score(career)
    traj_score = get_trajectory_score(career)
    loc_score = get_location_score(profile, signals)

    # Blend total YoE and AI-specific YoE
    combined_exp = 0.35 * exp_score + 0.65 * ai_exp_score

    # Weighted base score
    base = (
        0.22 * title_score +
        0.25 * combined_exp +
        0.20 * skills_score +
        0.22 * career_score +
        0.08 * traj_score +
        0.03 * loc_score
    )

    # Assessment bonus (platform-verified, small but signal)
    base += get_assessment_bonus(signals)

    # Education tier bonus (small)
    edu_tiers = {'tier_1': 0.015, 'tier_2': 0.008, 'tier_3': 0.004}
    if education:
        best_edu = max(edu_tiers.get(e.get('tier', 'unknown'), 0) for e in education)
        base += best_edu

    # Behavioral modifier (multiplicative)
    behavioral_mod = get_behavioral_modifier(signals)
    # Do NOT cap here — let raw scores differentiate; rescaling happens in main()
    final_score = base * behavioral_mod

    breakdown = {
        'honeypot': False,
        'title_score': title_score,
        'combined_exp': combined_exp,
        'skills_score': skills_score,
        'career_score': career_score,
        'traj_score': traj_score,
        'behavioral_mod': behavioral_mod,
        'top_skills': top_skills,
        'career_terms': career_terms,
    }
    return final_score, breakdown


# ============================================================================
# Reasoning Generation
# ============================================================================

def generate_reasoning(candidate: Dict, breakdown: Dict, rank: int) -> str:
    """
    Generate specific, grounded 1–2 sentence reasoning for each candidate.
    Only references data actually present in the candidate's profile.
    Penalized items (availability concerns, consulting background) are noted.
    """
    profile = candidate['profile']
    signals = candidate['redrob_signals']

    title = profile.get('current_title', 'Unknown')
    yoe = profile.get('years_of_experience', 0.0)
    location = profile.get('location', '')

    parts = []

    # Core identity
    parts.append(f"{title}, {yoe:.1f} yrs, {location}")

    # Career evidence (most important signal)
    career_terms = breakdown.get('career_terms', [])
    if career_terms:
        terms_str = ', '.join(career_terms[:3])
        parts.append(f"career evidence: {terms_str}")

    # Top relevant skills (from profile — not hallucinated)
    top_skills = breakdown.get('top_skills', [])
    if top_skills:
        parts.append(f"skills: {', '.join(top_skills[:3])}")

    # Availability signals
    open_w = signals.get('open_to_work_flag', False)
    rr = signals.get('recruiter_response_rate', 0.5)
    notice = signals.get('notice_period_days', 60)
    last_active = signals.get('last_active_date', '')

    if open_w:
        parts.append("open to work")

    if last_active:
        try:
            days_ago = (TODAY - date.fromisoformat(last_active)).days
            if days_ago > 180:
                parts.append(f"inactive {days_ago}d — availability risk")
            elif days_ago <= 14:
                parts.append("active this week")
        except ValueError:
            pass

    if rr < 0.15:
        parts.append(f"low response rate ({rr:.0%})")
    elif rr >= 0.75:
        parts.append(f"high response rate ({rr:.0%})")

    if notice <= 15:
        parts.append("immediately available")
    elif notice > 90:
        parts.append(f"{notice}d notice")

    return '; '.join(parts[:5]) + '.'


# ============================================================================
# Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Redrob Hackathon — Candidate Ranker',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--candidates', default='candidates.jsonl',
        help='Path to candidates JSONL file (plain or .gz)',
    )
    parser.add_argument(
        '--out', default='rajneel_chavan.csv',
        help='Output CSV path',
    )
    args = parser.parse_args()

    candidates_path = Path(args.candidates)
    if not candidates_path.exists():
        print(f"[ERROR] Candidates file not found: {candidates_path}", file=sys.stderr)
        sys.exit(1)

    print(f"[rank.py] Loading candidates from {candidates_path} ...")

    scored: List[Tuple[float, Dict, Dict]] = []
    n = 0

    # Support both plain JSONL and .gz
    if candidates_path.suffix == '.gz':
        import gzip
        opener = lambda: gzip.open(candidates_path, 'rt', encoding='utf-8')
    else:
        opener = lambda: open(candidates_path, 'r', encoding='utf-8')

    with opener() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                candidate = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[WARN] Skipping malformed line {n}: {e}", file=sys.stderr)
                continue

            score, breakdown = score_candidate(candidate)
            scored.append((score, candidate, breakdown))
            n += 1
            if n % 10_000 == 0:
                print(f"  ... {n:,} / 100,000 processed")

    print(f"[rank.py] Scored {n:,} candidates. Sorting ...")

    # Sort: score descending, tie-break candidate_id ascending (per spec)
    scored.sort(key=lambda x: (-x[0], x[1]['candidate_id']))

    top_100 = scored[:100]

    # Rescale top-100 scores to [0.10, 0.99] preserving relative order.
    # This prevents all top candidates from tying at a capped value and
    # makes the reasoning / score column meaningful to manual reviewers.
    raw_hi = top_100[0][0]
    raw_lo = top_100[-1][0]
    score_span = max(raw_hi - raw_lo, 1e-9)
    rescaled_100 = []
    for raw, cand, bd in top_100:
        rescaled = 0.10 + (raw - raw_lo) / score_span * (0.99 - 0.10)
        rescaled_100.append((rescaled, cand, bd))
    top_100 = rescaled_100

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, 'w', newline='', encoding='utf-8') as fh:
        writer = csv.writer(fh)
        writer.writerow(['candidate_id', 'rank', 'score', 'reasoning'])
        for rank, (score, candidate, breakdown) in enumerate(top_100, start=1):
            cid = candidate['candidate_id']
            reasoning = generate_reasoning(candidate, breakdown, rank)
            writer.writerow([cid, rank, f'{score:.4f}', reasoning])

    print(f"[rank.py] Submission written to {out_path}")
    print(f"\nTop 10:")
    for rank, (score, c, bd) in enumerate(top_100[:10], 1):
        p = c['profile']
        print(f"  {rank:2d}. {c['candidate_id']}  {p['current_title']:35s}  "
              f"{p['years_of_experience']:4.1f}yr  score={score:.4f}  "
              f"behavioral={bd.get('behavioral_mod', 0):.2f}")


if __name__ == '__main__':
    main()
