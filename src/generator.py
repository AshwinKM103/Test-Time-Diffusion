"""
Generator Module

Handles LLM interactions for the Test-Time-Diffusion framework.

The implementation supports:
    - Connecting to OpenRouter and local vLLM models based on hardware
    - Generating responses from the LLM given prompts
    - Robustly parsing self-evolution feedback responses
    - Implementing the Component-wise Self-Evolution algorithm

Key classes / functions:
    - get_openrouter_client: Retrieves the OpenRouter API client.
    - get_local_client: Retrieves a local vLLM client.
    - get_llm_response: Gets a response from the active LLM.
    - _parse_evolution_response: Parses structured self-evolution feedback.
    - self_evolve: Implements Component-wise Self-Evolution.

Version:
    - 05-Jun-2026 (Version 1.0): Initial implementation and documentation.
"""
import os
from openai import OpenAI
from dotenv import load_dotenv
from src.utils.logger import CustomLogger
logger = CustomLogger.setup_task_logger(__name__, output_dir="outputs/logs")
import torch

load_dotenv()


def get_openrouter_client():
    """
    Retrieves the OpenRouter API client.

    Configures the OpenAI client with OpenRouter's base URL, API key,
    and necessary headers (Referer and Title).

    Returns:
        OpenAI: The configured OpenRouter client.

    Example:
        >>> client = get_openrouter_client()
        >>> client.base_url
        'https://openrouter.ai/api/v1'
    """
    headers = {}
    if os.getenv("HTTP_REFERER"):
        headers["HTTP-Referer"] = os.getenv("HTTP_REFERER")
    if os.getenv("X_TITLE"):
        headers["X-Title"] = os.getenv("X_TITLE")

    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        default_headers=headers if headers else None,
    )


def get_local_client(port: int = 3002):
    """
    Retrieves a local vLLM client.

    Configures the OpenAI client to connect to a local vLLM server running
    on the specified port.

    Args:
        port (int): The port number where the local vLLM server is running. Default is 3002.

    Returns:
        OpenAI: The configured local vLLM client.

    Example:
        >>> client = get_local_client(port=3002)
    """
    return OpenAI(base_url=f"http://127.0.0.1:{port}/v1", api_key="None")


from src.config import config

provider = config.llm.provider
if provider == "auto":
    provider = "openrouter" if not torch.cuda.is_available() else "local"

if provider == "openrouter":
    client = get_openrouter_client()
    ACTIVE_MODEL = config.llm.openrouter_model
    logger.info(f"Using OpenRouter model: {ACTIVE_MODEL}")
else:
    client = get_local_client(port=config.llm.local_port)
    ACTIVE_MODEL = config.llm.local_model
    logger.info(f"Using local vLLM model: {ACTIVE_MODEL}")


def get_llm_response(
    prompt: str, system_prompt: str = "You are a world-class research assistant."
) -> str:
    """
    Gets a response from the active LLM given a prompt.

    Sends a chat completion request to the configured client using the
    specified system prompt and user prompt.

    Args:
        prompt (str): The user prompt to send to the LLM.
        system_prompt (str): The system prompt defining the LLM's role. Default is a world-class research assistant.

    Returns:
        str: The generated response from the LLM.

    Example:
        >>> response = get_llm_response("What is the capital of France?")
        >>> "Paris" in response
        True
    """
    resp = client.chat.completions.create(
        model=ACTIVE_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
    )
    content = resp.choices[0].message.content
    content = content if content else ""
    logger.debug(f"Prompt[:200]: {prompt[:200]}\nResponse[:200]: {content[:200]}")
    return content


def _parse_evolution_response(response: str, fallback: str) -> tuple[str, int, int]:
    """
    Robustly parse the structured self-evolution feedback response.

    Extracts the revised text, helpfulness score, and comprehensiveness score
    from the model's feedback response. Falls back to default values on any parsing failure.

    Args:
        response (str): The raw response string from the model containing the feedback.
        fallback (str): The fallback text to use if the revised text is missing or parsing fails.

    Returns:
        tuple[str, int, int]: A tuple containing the revised text, helpfulness score, and comprehensiveness score.

    Example:
        >>> resp = "HELPFULNESS_SCORE: 8\\nCOMPREHENSIVENESS_SCORE: 7\\nREVISED_TEXT: Improved text"
        >>> _parse_evolution_response(resp, "fallback")
        ('Improved text', 8, 7)
    """
    revised_text = fallback
    h_score, c_score = 5, 5

    try:
        if "REVISED_TEXT:" in response:
            revised_text = response.split("REVISED_TEXT:", 1)[1].strip()
        else:
            logger.warning("REVISED_TEXT marker missing in evolution response; using original variant.")

        if "HELPFULNESS_SCORE:" in response:
            raw_h = response.split("HELPFULNESS_SCORE:", 1)[1].split("\n")[0].strip()
            h_score = int(raw_h)

        if "COMPREHENSIVENESS_SCORE:" in response:
            raw_c = response.split("COMPREHENSIVENESS_SCORE:", 1)[1].split("\n")[0].strip()
            c_score = int(raw_c)

    except (ValueError, IndexError) as exc:
        logger.warning(f"Failed to parse evolution response ({exc}); using fallback values.")

    return revised_text, h_score, c_score


def self_evolve(
    initial_prompt: str,
    system_prompt: str,
    num_variants: int,
    evolution_steps: int,
) -> tuple[str, list[str]]:
    """
    Implements the Component-wise Self-Evolution algorithm from the TTD-DR paper.

    Generates multiple variants of a response to the initial prompt, then iteratively
    evaluates and refines them using a dual-criterion auto-rater (helpfulness and
    comprehensiveness). Finally, merges the best evolved variants into a single superior output.

    Args:
        initial_prompt (str): The initial prompt or request.
        system_prompt (str): The system prompt defining the LLM's role.
        num_variants (int): The number of distinct variants to generate initially.
        evolution_steps (int): The number of self-evolution iteration steps.

    Returns:
        tuple[str, list[str]]: A tuple containing the final merged text and the list of evolved variants.

    Example:
        >>> merged, variants = self_evolve("Explain quantum computing", "You are an expert", 2, 1)
    """
    # Step 1 — Initial States: generate diverse variants
    variants = [
        get_llm_response(initial_prompt, system_prompt) for _ in range(num_variants)
    ]

    for i in range(evolution_steps):
        scored_variants: list[tuple[str, float]] = []

        for variant in variants:
            # Dual-criterion auto-rater prompt (helpfulness + comprehensiveness)
            critique_prompt = f"""
You are an expert research evaluator. Evaluate the following text on two dimensions:

1. **Helpfulness** (1–10): Does it directly and completely address the original request?
2. **Comprehensiveness** (1–10): Does it cover all necessary aspects without major omissions?

Original Request:
{initial_prompt}

Text to Evaluate:
---
{variant}
---

Provide your evaluation in EXACTLY this format and nothing else:
HELPFULNESS_SCORE: [integer 1-10]
COMPREHENSIVENESS_SCORE: [integer 1-10]
CRITIQUE: [2-3 sentence critique addressing both dimensions]
REVISED_TEXT: [Improved version of the text that addresses all points in the critique]
"""
            feedback_response = get_llm_response(
                critique_prompt, "You are a critical and constructive reviewer."
            )

            revised_text, h_score, c_score = _parse_evolution_response(
                feedback_response, fallback=variant
            )
            combined_score = (h_score + c_score) / 2.0
            scored_variants.append((revised_text, combined_score))
            logger.debug(
                f"Evolution step {i+1}: helpfulness={h_score}, comprehensiveness={c_score}, "
                f"combined={combined_score:.1f}"
            )

        variants = [text for text, _ in scored_variants]

    # Step 2 — Cross-over (Merge): combine all evolved variants into a single superior output
    if len(variants) == 1:
        # Skip merge call when there is only one variant to avoid unnecessary tokens
        return variants[0], variants

    merge_prompt = f"""
You are given several refined texts that all attempt to answer an original request.
Synthesize them into a single, comprehensive, and superior final text that combines
the strongest elements of each.

Original Request:
{initial_prompt}

Refined Texts to Merge:
---
{"---".join(variants)}
---

Produce the final, merged text.
"""
    final_merged_text = get_llm_response(merge_prompt, system_prompt)
    return final_merged_text, variants
