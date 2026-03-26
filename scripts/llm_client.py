"""
bmad-dl LLM client — thin, config-driven wrapper.

Supports:
  - Anthropic (claude-*): uses anthropic SDK
  - Any OpenAI-compatible endpoint (OpenAI, Ollama, vLLM, Together, Groq, Mistral…):
    uses openai SDK with configurable base_url

Config is read from configs/llm_config.yaml in the project root.
The config file is created by init_project.py (bmad-dl-scaffold).

Usage
-----
  from scripts.llm_client import LLMClient

  client = LLMClient()                          # reads configs/llm_config.yaml
  response = client.chat("Summarise this text: ...")
  print(response)

  # Or pass raw messages
  response = client.chat([
      {"role": "system", "content": "You are a data scientist."},
      {"role": "user", "content": "What does this metric mean?"},
  ])
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Union

import yaml

# Default config path — relative to the project root where scripts are run from
DEFAULT_CONFIG_PATH = "configs/llm_config.yaml"


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(config_path: str = DEFAULT_CONFIG_PATH) -> dict:
    """Load llm_config.yaml. Falls back to env-var defaults if file is missing."""
    path = Path(config_path)
    if path.exists():
        with path.open() as f:
            cfg = yaml.safe_load(f) or {}
    else:
        cfg = {}

    # Fill defaults
    cfg.setdefault("provider", os.getenv("BMAD_LLM_PROVIDER", "anthropic"))
    cfg.setdefault("model", os.getenv("BMAD_LLM_MODEL", "claude-sonnet-4-6"))
    cfg.setdefault("base_url", os.getenv("BMAD_LLM_BASE_URL"))          # None for Anthropic
    cfg.setdefault("api_key_env", os.getenv("BMAD_LLM_API_KEY_ENV", "ANTHROPIC_API_KEY"))
    cfg.setdefault("max_tokens", int(os.getenv("BMAD_LLM_MAX_TOKENS", "8192")))
    cfg.setdefault("temperature", float(os.getenv("BMAD_LLM_TEMPERATURE", "0.0")))

    return cfg


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class LLMClient:
    """
    Provider-agnostic LLM client.

    Parameters
    ----------
    config_path:
        Path to llm_config.yaml. Default: configs/llm_config.yaml
    config:
        Pass a dict directly to override file-based config entirely.
    """

    def __init__(
        self,
        config_path: str = DEFAULT_CONFIG_PATH,
        config: dict | None = None,
    ) -> None:
        self.cfg = config if config is not None else load_config(config_path)
        self._client = self._build_client()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chat(
        self,
        messages: Union[str, list[dict]],
        *,
        system: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        """
        Send a chat request and return the assistant's text response.

        Parameters
        ----------
        messages:
            Either a plain string (treated as a single user message) or a
            list of {"role": ..., "content": ...} dicts.
        system:
            Optional system prompt (merged into messages list for OpenAI
            providers; passed as the `system` param for Anthropic).
        max_tokens / temperature:
            Per-call overrides; fall back to config values.
        """
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]

        max_tok = max_tokens or self.cfg["max_tokens"]
        temp = temperature if temperature is not None else self.cfg["temperature"]

        if self.cfg["provider"] == "anthropic":
            return self._call_anthropic(messages, system=system, max_tokens=max_tok, temperature=temp)
        else:
            return self._call_openai_compatible(messages, system=system, max_tokens=max_tok, temperature=temp)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_client(self):
        api_key = os.getenv(self.cfg["api_key_env"])
        if not api_key:
            raise EnvironmentError(
                f"API key not found. Expected env var: {self.cfg['api_key_env']}. "
                f"Set it or update api_key_env in configs/llm_config.yaml."
            )

        if self.cfg["provider"] == "anthropic":
            try:
                import anthropic as _anthropic
            except ImportError:
                raise ImportError(
                    "anthropic package not installed. Run: uv add anthropic"
                )
            return _anthropic.Anthropic(api_key=api_key)
        else:
            try:
                from openai import OpenAI
            except ImportError:
                raise ImportError(
                    "openai package not installed. Run: uv add openai"
                )
            kwargs: dict = {"api_key": api_key}
            if self.cfg.get("base_url"):
                kwargs["base_url"] = self.cfg["base_url"]
            return OpenAI(**kwargs)

    def _call_anthropic(
        self,
        messages: list[dict],
        *,
        system: str | None,
        max_tokens: int,
        temperature: float,
    ) -> str:
        kwargs: dict = dict(
            model=self.cfg["model"],
            max_tokens=max_tokens,
            temperature=temperature,
            messages=messages,
        )
        if system:
            kwargs["system"] = system

        response = self._client.messages.create(**kwargs)
        return response.content[0].text

    def _call_openai_compatible(
        self,
        messages: list[dict],
        *,
        system: str | None,
        max_tokens: int,
        temperature: float,
    ) -> str:
        full_messages = []
        if system:
            full_messages.append({"role": "system", "content": system})
        full_messages.extend(messages)

        response = self._client.chat.completions.create(
            model=self.cfg["model"],
            messages=full_messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content
