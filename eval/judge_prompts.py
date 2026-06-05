"""
Judge prompts for TTD-DR evaluation.

These prompts are used by the LLM-as-judge evaluation scripts to score
generated research reports against reference queries. They implement the
helpfulness + comprehensiveness dual-criterion auto-rater style described
in the TTD-DR paper.

Evaluation workflow:
  1. Generate reports with the pipeline (eval_hle.py / eval_gaia.py).
  2. Store reports as JSON files.
  3. Run the judge prompts against stored reports at a later time.
"""

# ---------------------------------------------------------------------------
# Helpfulness judge
# ---------------------------------------------------------------------------
HELPFULNESS_JUDGE_PROMPT = """
You are an expert research quality evaluator.

**Evaluation Criterion: Helpfulness**
Helpfulness measures how directly, accurately, and completely the research
report addresses the user's original query.

**User Query:**
{query}

**Research Report to Evaluate:**
{report}

Rate the helpfulness of this report on a scale from 1 to 10:
  10 = Directly and completely answers the query with high accuracy.
   7 = Mostly answers the query with only minor gaps.
   4 = Partially answers the query with significant omissions or inaccuracies.
   1 = Does not answer the query at all.

Respond in EXACTLY this format and nothing else:
HELPFULNESS_SCORE: [integer 1-10]
HELPFULNESS_RATIONALE: [1-2 sentence justification]
"""

# ---------------------------------------------------------------------------
# Comprehensiveness judge
# ---------------------------------------------------------------------------
COMPREHENSIVENESS_JUDGE_PROMPT = """
You are an expert research quality evaluator.

**Evaluation Criterion: Comprehensiveness**
Comprehensiveness measures how thoroughly the research report covers all
relevant aspects of the query without major topic omissions.

**User Query:**
{query}

**Research Report to Evaluate:**
{report}

Rate the comprehensiveness of this report on a scale from 1 to 10:
  10 = Covers all key aspects in appropriate depth; no important topics missing.
   7 = Covers most aspects with only minor omissions.
   4 = Covers the topic superficially; several important aspects are missing.
   1 = Covers almost nothing relevant to the query.

Respond in EXACTLY this format and nothing else:
COMPREHENSIVENESS_SCORE: [integer 1-10]
COMPREHENSIVENESS_RATIONALE: [1-2 sentence justification]
"""

# ---------------------------------------------------------------------------
# Combined (single-call) judge — used when token budget is tight
# ---------------------------------------------------------------------------
COMBINED_JUDGE_PROMPT = """
You are an expert research quality evaluator.

Evaluate the following research report on two dimensions:

1. **Helpfulness** (1–10): How directly and accurately does the report answer the query?
2. **Comprehensiveness** (1–10): How thoroughly does it cover all relevant aspects?

**User Query:**
{query}

**Research Report to Evaluate:**
{report}

Respond in EXACTLY this format and nothing else:
HELPFULNESS_SCORE: [integer 1-10]
HELPFULNESS_RATIONALE: [1-2 sentence justification]
COMPREHENSIVENESS_SCORE: [integer 1-10]
COMPREHENSIVENESS_RATIONALE: [1-2 sentence justification]
OVERALL_SCORE: [average of the two scores, to 1 decimal place]
"""

# ---------------------------------------------------------------------------
# Correctness judge (used for HLE / GAIA where a ground-truth answer exists)
# ---------------------------------------------------------------------------
CORRECTNESS_JUDGE_PROMPT = """
You are a strict grading assistant.

**Question:**
{question}

**Ground-Truth Answer:**
{ground_truth}

**Model's Generated Answer:**
{generated_answer}

Does the generated answer correctly answer the question, taking the ground-truth as the reference?

Respond in EXACTLY this format and nothing else:
GRADE: [C for Correct, I for Incorrect]
RATIONALE: [1 sentence explanation]
"""


def parse_judge_scores(response: str) -> dict:
    """
    Parse a judge response into a dict of score fields.

    Extracts HELPFULNESS_SCORE, COMPREHENSIVENESS_SCORE, OVERALL_SCORE,
    GRADE, and rationales from the judge's text output. Falls back to None
    for any field that cannot be parsed.

    Args:
        response (str): The raw text response from the judge LLM.

    Returns:
        dict: A dictionary containing the parsed scores and rationales.

    Example:
        >>> text = "HELPFULNESS_SCORE: 8\\nHELPFULNESS_RATIONALE: Good detail."
        >>> scores = parse_judge_scores(text)
        >>> scores["HELPFULNESS_SCORE"]
        8
    """
    result = {}
    int_fields = ["HELPFULNESS_SCORE", "COMPREHENSIVENESS_SCORE"]
    float_fields = ["OVERALL_SCORE"]
    str_fields = ["GRADE", "HELPFULNESS_RATIONALE", "COMPREHENSIVENESS_RATIONALE", "RATIONALE"]

    for field in int_fields:
        if f"{field}:" in response:
            try:
                raw = response.split(f"{field}:", 1)[1].split("\n")[0].strip()
                result[field] = int(raw)
            except (ValueError, IndexError):
                result[field] = None

    for field in float_fields:
        if f"{field}:" in response:
            try:
                raw = response.split(f"{field}:", 1)[1].split("\n")[0].strip()
                result[field] = float(raw)
            except (ValueError, IndexError):
                result[field] = None

    for field in str_fields:
        if f"{field}:" in response:
            try:
                result[field] = response.split(f"{field}:", 1)[1].split("\n")[0].strip()
            except IndexError:
                result[field] = None

    return result
