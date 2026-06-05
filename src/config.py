import yaml
from pathlib import Path
from pydantic import BaseModel

class PipelineConfig(BaseModel):
    max_search_iterations: int = 20
    num_search_query_variants: int = 5
    report_num_variants: int = 1
    report_evolution_steps: int = 1
    n_min_before_exit: int = 5
    num_variants: int = 1
    evolution_steps: int = 1
    search_top_k: int = 50

class LLMConfig(BaseModel):
    provider: str = "auto"
    local_model: str = "Qwen/Qwen3-4B-Instruct-2507"
    local_port: int = 3002
    openrouter_model: str = "alibaba/tongyi-deepresearch-30b-a3b:free"

class RetrievalConfig(BaseModel):
    backend: str = "tavily"
    reranker_model: str = "tomaarsen/Qwen3-Reranker-0.6B-seq-cls"

class EvalConfig(BaseModel):
    gaia_output_dir: str = "eval/results/gaia"
    hle_output_dir: str = "eval/results/hle"
    num_samples: int = 20
    max_iterations: int = 20
    gaia_level: int | None = None

class AppConfig(BaseModel):
    pipeline: PipelineConfig
    llm: LLMConfig
    retrieval: RetrievalConfig
    eval: EvalConfig

def load_config() -> AppConfig:
    """Loads and validates the configuration from configs/config.yaml using Pydantic."""
    config_path = Path(__file__).parent.parent / "configs" / "config.yaml"
    try:
        with open(config_path, "r") as f:
            data = yaml.safe_load(f)
        return AppConfig(**data)
    except FileNotFoundError:
        # Provide fallback if config file is missing
        return AppConfig(
            pipeline=PipelineConfig(),
            llm=LLMConfig(),
            retrieval=RetrievalConfig(),
            eval=EvalConfig()
        )

# Global configuration instance
config = load_config()
