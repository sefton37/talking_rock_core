from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    val = raw.strip().lower()
    if val in {"1", "true", "yes", "y", "on"}:
        return True
    if val in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _env_bool2(primary: str, fallback: str, default: bool = False) -> bool:
    """Read a bool env var with a primary name and a legacy fallback name."""
    if os.environ.get(primary) is not None:
        return _env_bool(primary, default)
    return _env_bool(fallback, default)


@dataclass(frozen=True)
class Settings:
    """Static settings for the local service.

    Keep defaults local and auditable; no network endpoints beyond localhost.
    """

    root_dir: Path = Path(__file__).resolve().parents[2]
    # TALKINGROCK_DATA_DIR takes precedence; REOS_DATA_DIR is the legacy fallback
    data_dir: Path = Path(
        os.environ.get("TALKINGROCK_DATA_DIR")
        or os.environ.get("REOS_DATA_DIR")
        or str(Path.home() / ".talkingrock")
    )
    events_path: Path = data_dir / "events.jsonl"
    audit_path: Path = data_dir / "audit.log"
    log_path: Path = data_dir / "talkingrock.log"
    log_level: str = (
        os.environ.get("TALKINGROCK_LOG_LEVEL")
        or os.environ.get("REOS_LOG_LEVEL")
        or "INFO"
    )
    log_max_bytes: int = int(
        os.environ.get("TALKINGROCK_LOG_MAX_BYTES")
        or os.environ.get("REOS_LOG_MAX_BYTES")
        or str(1_000_000)
    )
    log_backup_count: int = int(
        os.environ.get("TALKINGROCK_LOG_BACKUP_COUNT")
        or os.environ.get("REOS_LOG_BACKUP_COUNT")
        or "3"
    )
    host: str = (
        os.environ.get("TALKINGROCK_HOST")
        or os.environ.get("REOS_HOST")
        or "127.0.0.1"
    )
    port: int = int(
        os.environ.get("TALKINGROCK_PORT")
        or os.environ.get("REOS_PORT")
        or "8010"
    )
    ollama_url: str = (
        os.environ.get("TALKINGROCK_OLLAMA_URL")
        or os.environ.get("REOS_OLLAMA_URL")
        or "http://127.0.0.1:11434"
    )

    def __post_init__(self) -> None:
        """Validate settings that must be constrained for zero-trust."""
        from urllib.parse import urlparse
        parsed = urlparse(self.ollama_url)
        if parsed.hostname not in ("localhost", "127.0.0.1", "::1", None):
            raise ValueError(
                f"TALKINGROCK_OLLAMA_URL must point to localhost (got {parsed.hostname!r}). "
                "Talking Rock is local-only — remote LLM endpoints are not allowed."
            )
    ollama_model: str | None = (
        os.environ.get("TALKINGROCK_OLLAMA_MODEL")
        or os.environ.get("REOS_OLLAMA_MODEL")
        or None
    )

    # =========================================================================
    # Git Integration (OPTIONAL - M5 Roadmap Feature)
    # =========================================================================
    # Git integration is DISABLED by default. Core functionality
    # (natural language Linux control) does NOT depend on git features.
    #
    # When enabled, projects can:
    # - Analyze code changes vs project roadmap/charter
    # - Provide commit review and suggestions
    # - Track alignment with project goals
    #
    # Enable via: TALKINGROCK_GIT_INTEGRATION_ENABLED=true
    # =========================================================================
    git_integration_enabled: bool = _env_bool2(
        "TALKINGROCK_GIT_INTEGRATION_ENABLED", "REOS_GIT_INTEGRATION_ENABLED", False
    )

    # Commit code review (requires git_integration_enabled).
    # When enabled, the agent will read commit patches via `git show` and send them to the local LLM.
    auto_review_commits: bool = _env_bool2(
        "TALKINGROCK_AUTO_REVIEW_COMMITS", "REOS_AUTO_REVIEW_COMMITS", False
    )
    auto_review_commits_include_diff: bool = _env_bool2(
        "TALKINGROCK_AUTO_REVIEW_COMMITS_INCLUDE_DIFF",
        "REOS_AUTO_REVIEW_COMMITS_INCLUDE_DIFF",
        False,
    )
    auto_review_commits_cooldown_seconds: int = int(
        os.environ.get("TALKINGROCK_AUTO_REVIEW_COMMITS_COOLDOWN_SECONDS")
        or os.environ.get("REOS_AUTO_REVIEW_COMMITS_COOLDOWN_SECONDS")
        or "5"
    )

    # Git companion: which repo to observe (requires git_integration_enabled).
    # If unset, falls back to the workspace root if it's a git repo.
    repo_path: Path | None = (
        Path(
            os.environ.get("TALKINGROCK_REPO_PATH")
            or os.environ["REOS_REPO_PATH"]
        )
        if (os.environ.get("TALKINGROCK_REPO_PATH") or os.environ.get("REOS_REPO_PATH"))
        else None
    )

    # LLM context budgeting (heuristic, used for triggering reviews before overflow).
    llm_context_tokens: int = int(
        os.environ.get("TALKINGROCK_LLM_CONTEXT_TOKENS")
        or os.environ.get("REOS_LLM_CONTEXT_TOKENS")
        or "8192"
    )
    review_trigger_ratio: float = float(
        os.environ.get("TALKINGROCK_REVIEW_TRIGGER_RATIO")
        or os.environ.get("REOS_REVIEW_TRIGGER_RATIO")
        or "0.8"
    )
    review_trigger_cooldown_minutes: int = int(
        os.environ.get("TALKINGROCK_REVIEW_TRIGGER_COOLDOWN_MINUTES")
        or os.environ.get("REOS_REVIEW_TRIGGER_COOLDOWN_MINUTES")
        or "15"
    )

    # Estimation knobs (heuristics): how large changes feel in-context.
    review_overhead_tokens: int = int(
        os.environ.get("TALKINGROCK_REVIEW_OVERHEAD_TOKENS")
        or os.environ.get("REOS_REVIEW_OVERHEAD_TOKENS")
        or "800"
    )
    tokens_per_changed_line: int = int(
        os.environ.get("TALKINGROCK_TOKENS_PER_CHANGED_LINE")
        or os.environ.get("REOS_TOKENS_PER_CHANGED_LINE")
        or "6"
    )
    tokens_per_changed_file: int = int(
        os.environ.get("TALKINGROCK_TOKENS_PER_CHANGED_FILE")
        or os.environ.get("REOS_TOKENS_PER_CHANGED_FILE")
        or "40"
    )

    # =========================================================================
    # LSP Integration (Language Server Protocol)
    # =========================================================================
    # LSP provides real-time code intelligence: type errors, go-to-definition,
    # find-references, hover documentation without running tests.
    #
    # Requires language servers to be installed:
    # - Python: npm install -g pyright
    # - TypeScript: npm install -g typescript-language-server typescript
    # - Rust: rustup component add rust-analyzer
    # =========================================================================
    lsp_enabled: bool = _env_bool2("TALKINGROCK_LSP_ENABLED", "REOS_LSP_ENABLED", True)
    lsp_python_server: str = (
        os.environ.get("TALKINGROCK_LSP_PYTHON")
        or os.environ.get("REOS_LSP_PYTHON")
        or "pyright-langserver"
    )
    lsp_typescript_server: str = (
        os.environ.get("TALKINGROCK_LSP_TS")
        or os.environ.get("REOS_LSP_TS")
        or "typescript-language-server"
    )
    lsp_rust_server: str = (
        os.environ.get("TALKINGROCK_LSP_RUST")
        or os.environ.get("REOS_LSP_RUST")
        or "rust-analyzer"
    )
    lsp_startup_timeout: int = int(
        os.environ.get("TALKINGROCK_LSP_TIMEOUT")
        or os.environ.get("REOS_LSP_TIMEOUT")
        or "30"
    )


settings = Settings()

# Ensure data directories exist at import time (local-only side effect).
# Use 0o700 to prevent group/other access to user data.
settings.data_dir.mkdir(parents=True, exist_ok=True)
settings.data_dir.chmod(0o700)
