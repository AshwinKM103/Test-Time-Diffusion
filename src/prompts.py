"""
Prompts Module

Contains all LLM prompts used across the Test-Time-Diffusion RAG pipeline.
"""

PLAN_PROMPT = """
Based on the user's query, create a structured research plan.
This plan should outline the key areas, questions, and topics to investigate to provide a comprehensive answer.
The plan will serve as a scaffold for the entire research process.
Break it down into a list of concise points.

User Query: "{query}"
"""

INITIAL_DRAFT_PROMPT = """
Based on your internal knowledge and the user's query, write a preliminary, high-level draft report.
This draft will be refined later with retrieved information. It serves as a starting point and a "noisy" skeleton.

User Query: "{query}"
"""

SEARCH_QUERY_CANDIDATES_PROMPT = """
You are a researcher in an iterative process. Your goal is to formulate the next best search queries to gather information to refine an evolving research report.

**User's Original Query:**
{query}

**Overall Research Plan:**
{plan}

**Current Draft Report (State to be improved):**
{draft}

**History of Previous Searches (Queries and Answers):**
{history}

Generate exactly {n} distinct candidate search queries. Each should be concise, targeted, and aimed at filling different gaps in the current draft.
Do NOT repeat queries that are already present in the history.

Output ONLY the queries, one per line, numbered 1 through {n}. No preamble, no explanation.
"""

SEARCH_NOVELTY_SCORE_PROMPT = """
You are evaluating candidate search queries for a research task.

**Research History (queries already answered):**
{history}

**Research Plan (topics that still need coverage):**
{plan}

**Candidate Query:**
"{candidate}"

Score this candidate query on two dimensions (1–10 each):
- **Novelty** (1–10): How different is this from queries already in the history? 10 = completely new angle; 1 = near-duplicate.
- **Coverage** (1–10): How well does answering this query fill gaps identified in the research plan? 10 = directly addresses a major uncovered area.

Respond in EXACTLY this format and nothing else:
NOVELTY_SCORE: [integer 1-10]
COVERAGE_SCORE: [integer 1-10]
"""

ANSWER_SYNTHESIS_PROMPT = """
You have been given a search query and a list of retrieved documents.
Your task is to synthesize the information from these documents to provide a direct and comprehensive answer to the search query.
Focus only on the information present in the documents. Cite which document urls are relevant.

**Search Query:**
{search_query}

**Retrieved Document Chunks:**
{documents}

Synthesized Answer:
"""

DRAFT_REVISION_PROMPT = """
You are refining a research report. You have a previous version of the draft and new information from a recent search.
Your task is to integrate the new information into the draft to "denoise" it, making it more accurate, detailed, and comprehensive.
You can add new sections, expand existing points, or correct inaccuracies.

**User's Original Query:**
{query}

**Previous Draft Report:**
---
{draft}
---

**Newly Synthesized Information (from query: "{search_query}"):**
---
{new_answer}
---

Produce the new, revised draft report.
"""

EXIT_LOOP_PROMPT = """
You are a research completeness judge. Assess whether the current draft
adequately covers the research plan and whether further searches are likely to
add meaningful new information.

**Research Plan:**
{plan}

**Current Draft:**
{draft}

**Search History (queries executed so far):**
{history}

Answer with exactly one word: CONTINUE if more research is needed, or EXIT if the draft is sufficiently complete.
Do not output anything else.
"""

FINAL_REPORT_PROMPT = """
You are a research assistant tasked with writing a final, comprehensive report.
All the necessary research, including planning, iterative searching, and information synthesis, has been completed.
Use all the provided information to construct a well-structured, coherent, and detailed final report that directly addresses the user's original query.

**User's Original Query:**
{query}

**Initial Research Plan:**
{plan}

**Final Revised Draft (Skeleton for the report):**
{draft}

**Full History of Questions and Synthesized Answers:**
{history}

**Citations (cite inline using [n] notation — e.g. [1], [2]):**
{citations_formatted}

Now, write the final, polished report.
- Start with a **"Final Answer:"** short paragraph summarising the key findings.
- Follow with detailed sections covering the research plan points.
- Embed inline citations as [n] wherever a specific fact is drawn from a source.
- End with a **"References"** section listing all cited sources as:
  [n] Title — URL
"""
