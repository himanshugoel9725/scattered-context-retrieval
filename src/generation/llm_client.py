"""Unified LLM API client with caching, retry, rate limiting, and cost tracking.

Supports OpenAI, Anthropic, Google, and Together/Groq for Llama models.
Every call is cached to disk — reruns cost $0.
"""

import logging
import time
from typing import Any

import tiktoken

from src.utils.cache import cached_call, cost_tracker, make_cache_key, get_cache
from src.utils.config import get_api_key, get_models_config

logger = logging.getLogger(__name__)

# Cost per 1M tokens (as of 2025)
COST_TABLE = {
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4.1-nano": {"input": 0.10, "output": 0.40},
    "claude-sonnet-4-20250514": {"input": 3.00, "output": 15.00},
    "gemini-2.5-pro": {"input": 1.25, "output": 5.00},
    "meta-llama/Llama-3-70b-chat-hf": {"input": 0.90, "output": 0.90},
    "meta-llama/Llama-3-8b-chat-hf": {"input": 0.20, "output": 0.20},
}


def count_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    """Count tokens using tiktoken (cl100k_base for most models)."""
    try:
        enc = tiktoken.encoding_for_model(model)
    except KeyError:
        enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))


def _call_openai(model: str, messages: list[dict], max_tokens: int,
                 temperature: float) -> dict[str, Any]:
    """Call OpenAI API."""
    import openai
    client = openai.OpenAI(api_key=get_api_key("openai"))
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    choice = response.choices[0]
    usage = response.usage
    return {
        "content": choice.message.content,
        "input_tokens": usage.prompt_tokens,
        "output_tokens": usage.completion_tokens,
        "model": model,
    }


def _call_anthropic(model: str, messages: list[dict], max_tokens: int,
                    temperature: float) -> dict[str, Any]:
    """Call Anthropic API."""
    import anthropic
    client = anthropic.Anthropic(api_key=get_api_key("anthropic"))
    # Anthropic uses a separate system param
    system_msg = ""
    user_msgs = []
    for m in messages:
        if m["role"] == "system":
            system_msg = m["content"]
        else:
            user_msgs.append(m)
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": user_msgs,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if system_msg:
        kwargs["system"] = system_msg
    response = client.messages.create(**kwargs)
    return {
        "content": response.content[0].text,
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
        "model": model,
    }


def _call_google(model: str, messages: list[dict], max_tokens: int,
                 temperature: float) -> dict[str, Any]:
    """Call Google Generative AI API."""
    import google.generativeai as genai
    genai.configure(api_key=get_api_key("google"))
    gmodel = genai.GenerativeModel(model)
    # Convert messages to Google format
    prompt_parts = []
    for m in messages:
        prompt_parts.append(m["content"])
    combined = "\n\n".join(prompt_parts)
    response = gmodel.generate_content(
        combined,
        generation_config=genai.GenerationConfig(
            max_output_tokens=max_tokens,
            temperature=temperature,
        ),
    )
    # Estimate tokens since Google doesn't always return usage
    input_tokens = count_tokens(combined)
    output_tokens = count_tokens(response.text) if response.text else 0
    return {
        "content": response.text if response.text else "",
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "model": model,
    }


def _call_together(model: str, messages: list[dict], max_tokens: int,
                   temperature: float) -> dict[str, Any]:
    """Call Together AI API (for Llama models)."""
    import openai
    client = openai.OpenAI(
        api_key=get_api_key("together"),
        base_url="https://api.together.xyz/v1",
    )
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    choice = response.choices[0]
    usage = response.usage
    return {
        "content": choice.message.content,
        "input_tokens": usage.prompt_tokens if usage else 0,
        "output_tokens": usage.completion_tokens if usage else 0,
        "model": model,
    }


PROVIDER_DISPATCH = {
    "openai": _call_openai,
    "anthropic": _call_anthropic,
    "google": _call_google,
    "together": _call_together,
    "groq": _call_together,  # Groq uses OpenAI-compatible API
}


def llm_call(
    provider: str,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int = 2048,
    temperature: float = 0.0,
    cache: bool = True,
    max_retries: int = 3,
) -> dict[str, Any]:
    """Make an LLM API call with caching and retry logic.

    Returns dict with keys: content, input_tokens, output_tokens, model
    """
    if provider not in PROVIDER_DISPATCH:
        raise ValueError(f"Unknown provider: {provider}. "
                         f"Available: {list(PROVIDER_DISPATCH.keys())}")

    # Check cache first
    if cache:
        cache_key = make_cache_key(
            "llm", provider=provider, model=model,
            messages=str(messages), max_tokens=max_tokens,
            temperature=temperature,
        )
        disk_cache = get_cache()
        cached_result = disk_cache.get(cache_key)
        if cached_result is not None:
            logger.debug("LLM cache hit: %s/%s", provider, model)
            return cached_result

    # Call with retries
    call_fn = PROVIDER_DISPATCH[provider]
    last_error = None
    for attempt in range(max_retries):
        try:
            result = call_fn(model, messages, max_tokens, temperature)
            # Track cost
            costs = COST_TABLE.get(model, {"input": 0, "output": 0})
            cost_tracker.record(
                model=model,
                input_tokens=result["input_tokens"],
                output_tokens=result["output_tokens"],
                cost_per_1m_input=costs["input"],
                cost_per_1m_output=costs["output"],
            )
            # Cache result
            if cache:
                disk_cache.set(cache_key, result)
            return result
        except Exception as e:
            last_error = e
            wait = 2 ** attempt
            logger.warning("LLM call failed (attempt %d/%d): %s. Retrying in %ds",
                           attempt + 1, max_retries, e, wait)
            time.sleep(wait)

    raise RuntimeError(f"LLM call failed after {max_retries} retries: {last_error}")


def generate(prompt: str, model: str | None = None, provider: str | None = None,
             system: str | None = None, **kwargs) -> str:
    """Convenience wrapper: returns just the generated text."""
    config = get_models_config()
    if model is None:
        gen_config = config["llm"]["generation"]["dev"]
        model = gen_config["model"]
        provider = gen_config["provider"]
    elif provider is None:
        # Infer provider from model name
        if model.startswith("gpt"):
            provider = "openai"
        elif model.startswith("claude"):
            provider = "anthropic"
        elif model.startswith("gemini"):
            provider = "google"
        else:
            provider = "together"

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    result = llm_call(provider=provider, model=model, messages=messages, **kwargs)
    return result["content"]
