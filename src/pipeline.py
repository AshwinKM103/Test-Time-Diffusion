"""
Pipeline Module

Orchestrates the Test-Time-Diffusion research and report generation pipeline.

The implementation supports:
    - Managing the full RAG lifecycle (planning, drafting, searching, synthesizing, revising)
    - Evaluating and selecting search queries based on novelty and coverage
    - Using an LLM-based judge to decide when to exit the search loop
    - Generating a final comprehensive report with inline citations

Key classes / functions:
    - TTD_DR_Pipeline: Main pipeline class orchestrating the research process.
    - run_rag_dynamic: Runs the pipeline with a streaming callback for dynamic updates.
    - run_rag_static: Runs the pipeline synchronously and returns the final report string.

Version:
    - 05-Jun-2026 (Version 1.0): Initial implementation and documentation.
"""
import os
from typing import Callable, Dict, Any, List
from .retriever import retrieve
from .generator import get_llm_response, self_evolve
from src.utils.logger import CustomLogger
logger = CustomLogger.setup_task_logger(__name__, output_dir="outputs/logs")


from src.config import config
from src.prompts import *

MAX_SEARCH_ITERATIONS = config.pipeline.max_search_iterations
NUM_SEARCH_QUERY_VARIANTS = config.pipeline.num_search_query_variants
REPORT_NUM_VARIANTS = config.pipeline.report_num_variants
REPORT_EVOLUTION_STEPS = config.pipeline.report_evolution_steps
N_MIN_BEFORE_EXIT = config.pipeline.n_min_before_exit
NUM_VARIANTS = config.pipeline.num_variants
EVOLUTION_STEPS = config.pipeline.evolution_steps
SEARCH_TOP_K = config.pipeline.search_top_k





class TTD_DR_Pipeline:
    """
    Main pipeline class orchestrating the research process.

    This class manages the internal state of the research process, including
    the plan, current draft, search history, and citations. It coordinates the
    iterative retrieval, synthesis, and self-evolution steps.

    Args:
        callback (Callable[[Dict[str, Any]], None]): A callback function used to stream updates.

    Example:
        >>> def callback(data): pass
        >>> pipeline = TTD_DR_Pipeline(callback)
    """
    def __init__(self, callback: Callable[[Dict[str, Any]], None]):
        """
        Initializes the pipeline with a given callback.

        Args:
            callback (Callable[[Dict[str, Any]], None]): A callback function used to stream updates.
        """
        self.callback = callback
        self.plan = ""
        self.draft = ""
        self.q_a_history: List[Dict[str, str]] = []
        self.intermediate_log: List[str] = []
        # Item 12: Citations are now stored as rich dicts instead of bare URL strings.
        self.citations: List[Dict[str, str]] = []

    def _send_update(
        self,
        step_description: str | None = None,
        *,
        is_intermediate: bool = True,
        final_report_chunk: str | None = None,
        citations: List[Dict[str, str]] | None = None,
        complete: bool = False,
    ) -> None:
        """
        Helper to send updates through the callback.

        Constructs a data dictionary with the current state of the pipeline
        and sends it using the registered callback function.

        Args:
            step_description (str | None): A description of the current step to append to the log. Default is None.
            is_intermediate (bool): Whether this is an intermediate step. Default is True.
            final_report_chunk (str | None): The final report text, if completed. Default is None.
            citations (List[Dict[str, str]] | None): A list of citation dictionaries. Default is None.
            complete (bool): Whether the pipeline has completed. Default is False.
        """
        if step_description:
            self.intermediate_log.append(step_description)

        steps_text = "|||---|||".join(self.intermediate_log)
        data: Dict[str, Any] = {
            "intermediate_steps": steps_text if steps_text else None,
            "final_report": final_report_chunk,
            "is_intermediate": is_intermediate,
            "complete": complete,
        }
        if citations:
            # Surface only the URLs for the API response, preserving the richer
            # internal representation in self.citations.
            data["citations"] = [c.get("url", "") for c in citations]

        self.callback(data)

    def generate_research_plan(self, query: str):
        """
        Generate initial research plan based on user query.

        Uses the LLM to evolve a structured research plan that outlines key areas
        and questions to investigate.

        Args:
            query (str): The initial user query.
        """
        self._send_update("Generating initial research plan...")
        plan_prompt = PLAN_PROMPT.format(query=query)
        plan_text, _ = self_evolve(
            plan_prompt,
            "You are a strategic research planner.",
            num_variants=NUM_VARIANTS,
            evolution_steps=EVOLUTION_STEPS,
        )
        self.plan = plan_text
        plan_desc = f"**Research Plan Generated:**\n{self.plan}"
        self.q_a_history.append({"description": plan_desc})
        self._send_update(plan_desc)

    def generate_initial_draft(self, query: str):
        """
        Generate initial draft from internal knowledge.

        Uses the LLM to write a preliminary, high-level draft report based
        solely on the user's query and the model's internal knowledge.

        Args:
            query (str): The initial user query.
        """
        self._send_update("Generating initial draft from internal knowledge...")
        draft_prompt = INITIAL_DRAFT_PROMPT.format(query=query)
        self.draft = get_llm_response(draft_prompt)
        draft_desc = f"**Initial Draft Created:**\n{self.draft[:200]}..."
        self.q_a_history.append({"description": draft_desc})
        self._send_update(draft_desc)

    def _score_query_candidate(self, candidate: str, history_str: str) -> float:
        """
        Score a single candidate search query for novelty and plan coverage.

        Uses the LLM to evaluate the candidate query against the search history
        and the research plan, returning a combined score.

        Args:
            candidate (str): The candidate search query to score.
            history_str (str): A formatted string of previously executed queries and answers.

        Returns:
            float: The combined score (novelty + coverage) / 2.
        """
        score_prompt = SEARCH_NOVELTY_SCORE_PROMPT.format(
            history=history_str, plan=self.plan, candidate=candidate
        )
        response = get_llm_response(score_prompt, "You are a query quality evaluator.")
        novelty, coverage = 5, 5
        try:
            if "NOVELTY_SCORE:" in response:
                novelty = int(response.split("NOVELTY_SCORE:", 1)[1].split("\n")[0].strip())
            if "COVERAGE_SCORE:" in response:
                coverage = int(response.split("COVERAGE_SCORE:", 1)[1].split("\n")[0].strip())
        except (ValueError, IndexError) as exc:
            logger.warning(f"Failed to parse query score: {exc}")
        return (novelty + coverage) / 2.0

    def generate_search_query(self, query: str, iteration: int, max_iterations: int) -> str:
        """
        Generate the next search query for the current iteration.

        Generates multiple candidate queries, scores each for novelty and plan coverage,
        rejects any already in the history, and selects the highest-scoring novel query.

        Args:
            query (str): The original user query.
            iteration (int): The current iteration number.
            max_iterations (int): The maximum number of search iterations.

        Returns:
            str: The selected search query.
        """
        step_desc = f"**Iteration {iteration + 1}/{max_iterations}:** Generating candidate search queries..."
        self._send_update(step_desc)

        history_str = "\n".join(
            [
                f"Q: {item['query']}\nA: {item['answer']}"
                for item in self.q_a_history
                if "query" in item
            ]
        )
        history_queries = {
            item["query"].strip().lower()
            for item in self.q_a_history
            if "query" in item
        }

        candidates_prompt = SEARCH_QUERY_CANDIDATES_PROMPT.format(
            query=query,
            plan=self.plan,
            draft=self.draft,
            history=history_str,
            n=NUM_SEARCH_QUERY_VARIANTS,
        )
        raw_candidates = get_llm_response(candidates_prompt)

        # Parse numbered list "1. ...", "2. ...", etc.
        candidates: List[str] = []
        for line in raw_candidates.strip().splitlines():
            line = line.strip()
            if not line:
                continue
            # Strip leading "1. " / "1) " numbering
            for sep in [". ", ") ", ": "]:
                if line[0].isdigit() and sep in line:
                    line = line.split(sep, 1)[1].strip()
                    break
            if line:
                candidates.append(line)

        if not candidates:
            logger.warning("No candidates parsed from LLM output; falling back to raw text.")
            candidates = [raw_candidates.strip()]

        # Filter queries already answered
        novel_candidates = [
            c for c in candidates if c.strip().lower() not in history_queries
        ]
        if not novel_candidates:
            logger.warning("All candidates are duplicates of history; using first candidate anyway.")
            novel_candidates = candidates[:1]

        # Score each novel candidate and pick the best
        best_query = novel_candidates[0]
        if len(novel_candidates) > 1:
            scores = [
                (cand, self._score_query_candidate(cand, history_str))
                for cand in novel_candidates
            ]
            scores.sort(key=lambda x: x[1], reverse=True)
            best_query, best_score = scores[0]
            logger.debug(
                f"Selected query (score={best_score:.1f}): {best_query!r}"
            )

        self._send_update(f"**Searching for:** `{best_query}`")
        return best_query

    def retrieve_and_synthesize_documents(
        self, search_query: str, iteration: int
    ) -> str:
        """
        Retrieve, rerank, and synthesize documents for the given search query.

        Fetches documents using the configured retrieval backend, chunks and reranks them,
        and then uses the LLM to synthesize a comprehensive answer based on the top chunks.

        Args:
            search_query (str): The specific search query to execute.
            iteration (int): The current iteration number (used for citation tracking).

        Returns:
            str: The synthesized answer based on the retrieved documents.
        """
        chunks = retrieve(search_query, top_k=SEARCH_TOP_K)
        chunks = chunks[:20]

        # Item 12: Build rich citation dicts preserving url, title, snippet, and context.
        new_citations: List[Dict[str, str]] = [
            {
                "url": doc.get("url", ""),
                "title": doc.get("title", ""),
                "snippet": doc.get("text", doc.get("content", ""))[:300],
                "search_query": search_query,
                "iteration": str(iteration + 1),
            }
            for doc in chunks
            if doc.get("url")
        ]
        self.citations.extend(new_citations)

        self._send_update(
            f"**Found {len(chunks)} documents.** Synthesizing answer...",
            citations=new_citations if new_citations else None,
        )

        doc_str = "\n\n".join(
            [f"ID: {doc['chunk_id']}\nURL: {doc.get('url','')}\nText: {doc['text']}..." for doc in chunks]
        )
        synth_prompt = ANSWER_SYNTHESIS_PROMPT.format(
            search_query=search_query, documents=doc_str
        )
        synthesized_answer, _ = self_evolve(
            synth_prompt,
            "You are a research analyst.",
            num_variants=NUM_VARIANTS,
            evolution_steps=EVOLUTION_STEPS,
        )
        self.q_a_history.append(
            {
                "description": f"**Synthesized Answer for `{search_query}`:**\n{synthesized_answer}",
                "query": search_query,
                "answer": synthesized_answer,
            }
        )
        self._send_update(
            f"**Synthesized Answer for `{search_query}`:**\n{synthesized_answer}"
        )
        return synthesized_answer

    def revise_draft_with_new_info(
        self, query: str, search_query: str, synthesized_answer: str, iteration: int
    ):
        """
        Revise the current draft with new synthesized information.

        Integrates the newly synthesized answer into the current draft, updating
        it to be more accurate and comprehensive.

        Args:
            query (str): The original user query.
            search_query (str): The search query that yielded the new information.
            synthesized_answer (str): The newly synthesized information to integrate.
            iteration (int): The current iteration number.
        """
        step_desc = "Revising draft with new information..."
        self._send_update(step_desc)
        revise_prompt = DRAFT_REVISION_PROMPT.format(
            query=query,
            draft=self.draft,
            search_query=search_query,
            new_answer=synthesized_answer,
        )
        self.draft = get_llm_response(revise_prompt)
        revised_desc = f"**Revised Draft {iteration + 1}:**\n{self.draft[:200]}..."
        self.q_a_history.append({"description": revised_desc})
        self._send_update(revised_desc)

    def check_exit_condition(self, iteration: int) -> bool:
        """
        LLM-based exit controller to determine if further searches are needed.

        Returns True (exit loop) when at least N_MIN_BEFORE_EXIT iterations have
        completed AND the LLM judge determines the draft is sufficiently complete.

        Args:
            iteration (int): The current iteration number.

        Returns:
            bool: True if the search loop should exit, False otherwise.
        """
        if iteration + 1 < N_MIN_BEFORE_EXIT:
            return False  # Too early to consider stopping

        history_str = "\n".join(
            item["query"] for item in self.q_a_history if "query" in item
        )
        prompt = EXIT_LOOP_PROMPT.format(
            plan=self.plan, draft=self.draft, history=history_str
        )
        verdict = get_llm_response(prompt, "You are a strict research completeness judge.").strip().upper()
        should_exit = verdict.startswith("EXIT")
        logger.info(
            f"Exit check at iteration {iteration + 1}: verdict={verdict!r}, exit={should_exit}"
        )
        return should_exit

    def perform_iterative_search_and_synthesis(self, query: str, max_iterations: int):
        """
        Perform iterative search and synthesis loop to refine the draft.

        Repeatedly generates search queries, retrieves and synthesizes information,
        and revises the draft until the exit condition is met or the maximum number
        of iterations is reached.

        Args:
            query (str): The original user query.
            max_iterations (int): The maximum number of search iterations allowed.
        """
        for i in range(max_iterations):
            search_query = self.generate_search_query(query, i, max_iterations)
            if not search_query or search_query.strip() == "":
                logger.warning("No valid search query generated; skipping iteration.")
                continue

            synthesized_answer = self.retrieve_and_synthesize_documents(search_query, i)
            self.revise_draft_with_new_info(query, search_query, synthesized_answer, i)

            # Item 2: Exit check (only activates after N_MIN_BEFORE_EXIT iterations)
            if self.check_exit_condition(i):
                self._send_update(
                    f"**Exit condition met at iteration {i + 1}.** Proceeding to final report."
                )
                break

    def generate_final_report(self, query: str):
        """
        Generate the final report using self-evolution.

        Formats the accumulated search history and citations, then uses the LLM
        to synthesize a polished, comprehensive final report with inline citations.

        Args:
            query (str): The original user query.
        """
        self._send_update("All research steps complete. Generating final report...")

        history_str = "\n\n".join(
            [
                f"**Question:** {item['query']}\n**Answer:** {item['answer']}"
                for item in self.q_a_history
                if "query" in item
            ]
        )

        # Item 12: Format citations as a numbered list so the model can embed [n] references.
        # Deduplicate by URL while preserving order.
        seen_urls: set[str] = set()
        unique_citations: List[Dict[str, str]] = []
        for c in self.citations:
            url = c.get("url", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                unique_citations.append(c)

        citations_formatted = "\n".join(
            f"[{idx + 1}] {c.get('title', 'No title')} — {c.get('url', '')}\n"
            f"    Snippet: \"{c.get('snippet', '')[:200]}\""
            for idx, c in enumerate(unique_citations)
        )

        final_prompt = FINAL_REPORT_PROMPT.format(
            query=query,
            plan=self.plan,
            draft=self.draft,
            history=history_str,
            citations_formatted=citations_formatted,
        )

        # Item 11: Apply self_evolve() to the final report generation.
        final_report_content, _ = self_evolve(
            final_prompt,
            "You are a world-class research writer.",
            num_variants=REPORT_NUM_VARIANTS,
            evolution_steps=REPORT_EVOLUTION_STEPS,
        )

        self._send_update(
            "Final report generated.",
            is_intermediate=False,
            final_report_chunk=final_report_content,
            citations=unique_citations,
            complete=True,
        )

    def run(self, query: str, max_iterations: int = MAX_SEARCH_ITERATIONS):
        """
        Executes the complete pipeline workflow.

        Coordinates plan generation, initial drafting, iterative search/synthesis,
        and final report generation.

        Args:
            query (str): The original user query.
            max_iterations (int): The maximum number of search iterations allowed. Default is MAX_SEARCH_ITERATIONS.
        """
        self.generate_research_plan(query)
        self.generate_initial_draft(query)
        self.perform_iterative_search_and_synthesis(query, max_iterations)
        self.generate_final_report(query)


def run_rag_dynamic(query: str, callback: Callable[[Dict[str, Any]], None]):
    """
    Runs the pipeline with a streaming callback for dynamic updates.

    Initializes the TTD_DR_Pipeline with the provided callback and executes
    the research process for the given query.

    Args:
        query (str): The initial user query to answer.
        callback (Callable[[Dict[str, Any]], None]): A callback function to receive intermediate and final state updates.

    Example:
        >>> run_rag_dynamic("How do transformers work?", lambda data: print("Update received"))
    """
    pipeline = TTD_DR_Pipeline(callback)
    pipeline.run(query)


def run_rag_static(query: str) -> str:
    """
    Runs the pipeline synchronously and returns the final report string.

    Initializes the TTD_DR_Pipeline with an internal callback that captures
    the final report once generation is complete, discarding intermediate updates.

    Args:
        query (str): The initial user query to answer.

    Returns:
        str: The fully generated and formatted final report.

    Example:
        >>> report = run_rag_static("Explain the theory of relativity.")
        >>> len(report) > 0
        True
    """
    final_report = ""

    def static_callback(data: Dict[str, Any]):
        nonlocal final_report
        if data["complete"]:
            final_report = data["final_report"]
        logger.debug(f"Static callback update: {data}")

    pipeline = TTD_DR_Pipeline(static_callback)
    pipeline.run(query)
    return final_report
