import hashlib
import json
import os
import threading
import time
from dataclasses import dataclass

from openai import OpenAI


@dataclass
class ChatResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    cached: bool = False


@dataclass
class Tracker:
    """Per-question cost accumulator (thread-local to each method call)."""
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def add(self, r: ChatResult):
        self.calls += 1
        self.prompt_tokens += r.prompt_tokens
        self.completion_tokens += r.completion_tokens

    def as_dict(self):
        return {"calls": self.calls,
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens}


class LLMClient:
    """OpenAI-compatible client (-> local vLLM) with a concurrent-safe on-disk cache."""

    def __init__(self, model, endpoint="http://localhost:8765/v1", api_key="EMPTY",
                 cache_dir="results/cache", use_cache=True, timeout=180, max_retries=5,
                 extra_body=None):
        self.model = model
        self.client = OpenAI(base_url=endpoint, api_key=api_key, timeout=timeout, max_retries=2)
        self.cache_dir = cache_dir
        self.use_cache = use_cache
        self.max_retries = max_retries
        self.extra_body = extra_body
        os.makedirs(cache_dir, exist_ok=True)
        self._lock = threading.Lock()
        self.global_calls = 0

    def _key(self, messages, temperature, max_tokens, top_p, seed):
        payload = json.dumps({"m": self.model, "msgs": messages, "t": temperature,
                              "mt": max_tokens, "tp": top_p, "seed": seed,
                              "extra_body": self.extra_body},
                             sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _path(self, key):
        return os.path.join(self.cache_dir, key + ".json")

    def chat(self, messages, temperature=0.0, max_tokens=1024, top_p=1.0, seed=None) -> ChatResult:
        key = self._key(messages, temperature, max_tokens, top_p, seed)
        path = self._path(key)
        if self.use_cache and os.path.exists(path):
            try:
                with open(path) as f:
                    d = json.load(f)
                return ChatResult(d["text"], d.get("prompt_tokens", 0),
                                  d.get("completion_tokens", 0), cached=True)
            except Exception:
                pass
        last_err = None
        for attempt in range(self.max_retries):
            try:
                kwargs = dict(model=self.model, messages=messages, temperature=temperature,
                              max_tokens=max_tokens, top_p=top_p)
                if seed is not None:
                    kwargs["seed"] = seed
                if self.extra_body:
                    kwargs["extra_body"] = self.extra_body
                resp = self.client.chat.completions.create(**kwargs)
                text = resp.choices[0].message.content or ""
                pt = resp.usage.prompt_tokens if resp.usage else 0
                ct = resp.usage.completion_tokens if resp.usage else 0
                if self.use_cache:
                    tmp = f"{path}.tmp.{os.getpid()}.{threading.get_ident()}"
                    with open(tmp, "w") as f:
                        json.dump({"text": text, "prompt_tokens": pt, "completion_tokens": ct}, f)
                    os.replace(tmp, path)
                with self._lock:
                    self.global_calls += 1
                return ChatResult(text, pt, ct, cached=False)
            except Exception as e:
                last_err = e
                time.sleep(min(2 ** attempt, 12))
        raise RuntimeError(f"LLM call failed after {self.max_retries} retries: {last_err}")


def ask(client, tr, messages, **kw) -> str:
    """Convenience: call chat, charge the tracker, return text."""
    r = client.chat(messages, **kw)
    if tr is not None:
        tr.add(r)
    return r.text
