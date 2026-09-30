import threading
import time
from collections import Counter

from openai import OpenAI


def _endpoint_for(model):
    """Provider that serves this model, from the shared registry.

    Routing is by model rather than by a module-level constant edited before
    each run. A constant is a silent-failure shape: pointing a run at an
    endpoint without the requested model produces empty replies and odd errors,
    which read as model behaviour rather than as a misrouted request.
    """
    import sys as _sys
    from pathlib import Path as _P
    _eval = str(_P(__file__).resolve().parent.parent / "evaluation")
    if _eval not in _sys.path:
        _sys.path.insert(0, _eval)
    from model_inventory import endpoint_for
    return endpoint_for(model)


# Kept for callers that still import it; the evaluation grid's provider.
BASE_URL = "https://api.zhizengzeng.com/v1"


class QuotaExhausted(RuntimeError):
    """The account's usage limit is spent.

    Distinct from an ordinary API error because it is terminal for the *run*,
    not just the call: every subsequent request fails the same way. A sweep that
    keeps going burns wall-clock (and, where billing is per request, money) to
    collect nothing. Callers should stop rather than continue.
    """


class OnlineLLM:
    def __init__(self, model, temperature, top_p, api_key, base_url=None):
        import sys as _sys
        from pathlib import Path as _P
        _eval = str(_P(__file__).resolve().parent.parent / "evaluation")
        if _eval not in _sys.path:
            _sys.path.insert(0, _eval)
        from model_inventory import (
            api_key_env_for,
            requested_model_for,
            thinking_mode_for,
        )

        # `model` is the experiment arm and therefore the result-directory key.
        # It can differ from the API model: the two DeepSeek V4 Flash arms both
        # request `deepseek-v4-flash` and differ only in the explicit thinking
        # switch. Keeping both names prevents their artifacts from colliding.
        self.model = model
        self.requested_model = requested_model_for(model)
        self.thinking_mode = thinking_mode_for(model)
        self.api_key_env = api_key_env_for(model)
        self.temperature = temperature
        self.top_p = top_p
        self.api_key = api_key
        # Routed by model unless the caller names an endpoint explicitly.
        self.base_url = base_url or _endpoint_for(model)

        # Provenance. The identifier the provider actually served can differ
        # from `requested_model` and is the only in-band evidence of which
        # weights answered. It is not recoverable after the fact, so record it
        # per call -- a run that did not capture it can never be traced back.
        self._prov_lock = threading.Lock()
        self.last_served_model = ""
        self.last_response_id = ""
        self.served_models = Counter()
        # Per-inference status.  Callers must not infer transport success from an
        # empty string: until now a five-attempt timeout and a model-authored
        # empty answer were indistinguishable and both were scored as bad code.
        self.last_error = ""
        self.last_failure_kind = ""
        self.last_attempts = 0
        self.last_attempt_elapsed_seconds = []
        self.last_finish_reason = ""
        self.last_reasoning_chars = 0
        self.last_completion_tokens = None
        self.last_reasoning_tokens = None

        # Wall-clock bound per HTTP request. Generous, because a reasoning
        # model with a large completion budget legitimately takes a while, but
        # finite so a stall cannot hold the sweep open indefinitely.
        import sys as _sys
        from pathlib import Path as _P
        _eval = str(_P(__file__).resolve().parent.parent / "evaluation")
        if _eval not in _sys.path:
            _sys.path.insert(0, _eval)
        from model_inventory import (
            inference_max_attempts_for,
            reasoning_effort_for,
            request_timeout_for,
            stream_response_for,
        )
        self.request_timeout = request_timeout_for(model)
        self.max_attempts = inference_max_attempts_for(model)
        self.reasoning_effort = reasoning_effort_for(model)
        self.stream_response = stream_response_for(model)

        # Set once if the provider refuses the requested temperature and
        # dictates its own. Sticky for the life of the client so the fallback
        # is paid for once rather than on every call.
        self._forced_temperature = None

    def reset_provenance(self):
        """Start a fresh accounting window (callers use one window per run)."""
        with self._prov_lock:
            self.last_served_model = ""
            self.last_response_id = ""
            self.served_models = Counter()

    def provenance(self):
        """Snapshot of what the provider served in the current window."""
        with self._prov_lock:
            return {
                "experiment_model": self.model,
                "requested_model": self.requested_model,
                "served_models": dict(self.served_models),
                "base_url": self.base_url,
                "api_key_env": self.api_key_env,
                "thinking_mode": self.thinking_mode,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "reasoning_effort": self.reasoning_effort,
                "stream": self.stream_response,
            }

    def infer(self, prompt, max_tokens):
        message = [
            {
                "role": "system",
                "content": "You are an expert in Datalog program synthesis."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
        base_url = self.base_url
        active_stream = [None]

        # Clear first: on a failure path we must not leave the previous call's
        # identifier standing, or the caller would attribute it to this one.
        with self._prov_lock:
            self.last_served_model = ""
            self.last_response_id = ""
        self.last_error = ""
        self.last_failure_kind = ""
        self.last_attempts = 0
        self.last_attempt_elapsed_seconds = []
        self.last_finish_reason = ""
        self.last_reasoning_chars = 0
        self.last_completion_tokens = None
        self.last_reasoning_tokens = None

        def call_api():
            client = OpenAI(
                api_key=self.api_key,
                base_url=base_url,
                # A real socket-level bound, so a hung request ends instead of
                # occupying a worker until someone kills the process.
                timeout=self.request_timeout,
                # This client must not retry on its own: `infer` already has a
                # retry loop, and the two multiply. Three SDK attempts inside
                # each of our attempts is latency nobody asked for and nobody
                # could see in the logs.
                max_retries=0,
            )

            req_kwargs = {
                "model": self.requested_model,
                "messages": message,
                "max_tokens": max_tokens,
            }
            req_kwargs.update(self._build_sampling_kwargs())
            extra_body = {}
            if self.thinking_mode:
                extra_body["thinking"] = {"type": self.thinking_mode}
            if self.reasoning_effort:
                # `extra_body` keeps this compatible with older OpenAI SDKs
                # that do not yet name DeepSeek's fields in their signature.
                # The fields are still serialized at the request body's top
                # level, exactly as the provider's OpenAI-format API expects.
                extra_body["reasoning_effort"] = self.reasoning_effort
            if extra_body:
                req_kwargs["extra_body"] = extra_body
            if self.stream_response:
                req_kwargs["stream"] = True
                req_kwargs["stream_options"] = {"include_usage": True}
            if self._forced_temperature is not None:
                req_kwargs["temperature"] = self._forced_temperature
                req_kwargs.pop("top_p", None)

            response = client.chat.completions.create(**req_kwargs)

            if self.stream_response:
                active_stream[0] = response
                try:
                    return self._consume_stream(response, max_tokens=max_tokens)
                finally:
                    active_stream[0] = None

            # Before the error check: a provider that errors still tells us
            # which model it routed to, and that is worth keeping.
            self._record_provenance(response)
            self._capture_response_metadata(response)

            error_message = self._extract_error_message(response)
            if error_message:
                raise RuntimeError(
                    f"Provider error for model={self.model}: {error_message}"
                )

            content = self._extract_text(response)
            if content:
                return content

            # A reasoning model can spend the whole budget on `reasoning_content`
            # and return empty `content` with finish_reason="length". That is a
            # budget problem, not a transient one: the same prompt under the same
            # cap truncates again, so it must be named and must not be retried.
            if self.last_finish_reason == "length":
                raise RuntimeError(
                    f"response truncated at max_tokens for model={self.model}: the model "
                    f"produced only reasoning within the {max_tokens}-token budget. "
                    "Raise max_tokens rather than retrying."
                )

            preview = self._preview_response(response)
            raise RuntimeError(
                f"Empty or unsupported response payload for model={self.model}. payload_preview={preview}"
            )

        tryCnt = 0
        while tryCnt < self.max_attempts:
            tryCnt += 1
            self.last_attempts = tryCnt
            attempt_started = time.monotonic()
            try:
                output = self.run_with_timeout(
                    call_api,
                    self.request_timeout + 10,
                    on_timeout=lambda: self._close_stream(active_stream[0]),
                )
                if output:
                    self.last_attempt_elapsed_seconds.append(
                        round(time.monotonic() - attempt_started, 3)
                    )
                    self.last_error = ""
                    self.last_failure_kind = ""
                    return output
            except Exception as e:
                self.last_attempt_elapsed_seconds.append(
                    round(time.monotonic() - attempt_started, 3)
                )
                self.last_error = str(e)
                self.last_failure_kind = self._failure_kind(e)
                if self._is_quota_error(e):
                    raise QuotaExhausted(
                        f"usage limit reached for model={self.model}; every further "
                        f"call will fail the same way. Original error: {e}"
                    ) from e
                # Some models accept only their default temperature. The probe
                # needs sampling spread, not a specific value, so retry once at
                # the value the provider will take -- and record it, because a
                # model sampled at a different temperature has a different noise
                # floor and its self-consistency is not directly comparable with
                # the others'.
                if self._is_temperature_error(e) and self._forced_temperature is None:
                    self._forced_temperature = 1.0
                    print(f"[TEMP] {self.model}: provider refuses temperature="
                          f"{self.temperature}; retrying at 1.0 and recording it")
                    continue
                print(f"API error: {e}")
                if not self._should_retry(e):
                    break
            # No unconditional sleep after the final attempt: it cannot help a
            # request that will never be made, and accumulates across a sweep.
            if tryCnt < self.max_attempts:
                time.sleep(2)

        if not self.last_error:
            self.last_error = "API returned no usable content"
            self.last_failure_kind = "empty_response"
        return ""

    @staticmethod
    def _close_stream(stream):
        if stream is None:
            return
        close = getattr(stream, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass

    def _consume_stream(self, response, max_tokens):
        """Collect visible text while retaining only aggregate CoT metadata.

        DeepSeek thinking responses emit reasoning deltas and keep-alives long
        before the final answer. Consuming the stream prevents that healthy
        progress from being mistaken for an idle socket. The chain of thought
        itself is deliberately neither returned nor persisted.
        """
        content = []
        served_model = ""
        response_id = ""
        finish_reason = ""
        reasoning_chars = 0
        completion_tokens = None
        reasoning_tokens = None

        try:
            for chunk in response:
                served_model = self._response_field(chunk, "model") or served_model
                response_id = self._response_field(chunk, "id") or response_id
                choices = getattr(chunk, "choices", None)
                if choices is None and isinstance(chunk, dict):
                    choices = chunk.get("choices")
                if choices:
                    first = choices[0]
                    delta = getattr(first, "delta", None)
                    if delta is None and isinstance(first, dict):
                        delta = first.get("delta")
                    if delta is not None:
                        visible = getattr(delta, "content", None)
                        reasoning = getattr(delta, "reasoning_content", None)
                        if isinstance(delta, dict):
                            visible = delta.get("content", visible)
                            reasoning = delta.get("reasoning_content", reasoning)
                        if isinstance(visible, str) and visible:
                            content.append(visible)
                        if isinstance(reasoning, str):
                            reasoning_chars += len(reasoning)
                    fr = getattr(first, "finish_reason", None)
                    if fr is None and isinstance(first, dict):
                        fr = first.get("finish_reason")
                    if isinstance(fr, str) and fr:
                        finish_reason = fr

                usage = getattr(chunk, "usage", None)
                if usage is None and isinstance(chunk, dict):
                    usage = chunk.get("usage")
                if usage:
                    completion_tokens = self._object_field(
                        usage, "completion_tokens", completion_tokens
                    )
                    details = self._object_field(usage, "completion_tokens_details", None)
                    if details:
                        reasoning_tokens = self._object_field(
                            details, "reasoning_tokens", reasoning_tokens
                        )
        finally:
            if served_model or response_id:
                self._record_provenance({"model": served_model, "id": response_id})
            self.last_finish_reason = finish_reason
            self.last_reasoning_chars = reasoning_chars
            self.last_completion_tokens = completion_tokens
            self.last_reasoning_tokens = reasoning_tokens

        text = "".join(content).strip()
        if text:
            return text
        if finish_reason == "length":
            raise RuntimeError(
                f"response truncated at max_tokens for model={self.model}: the model "
                f"produced only reasoning within the {max_tokens}-token budget. "
                "Raise max_tokens rather than retrying."
            )
        raise RuntimeError(
            f"Empty streamed response for model={self.model}; "
            f"finish_reason={finish_reason or 'unreported'}"
        )

    def _capture_response_metadata(self, response):
        self.last_finish_reason = self._finish_reason(response)
        choices = getattr(response, "choices", None)
        if choices is None and isinstance(response, dict):
            choices = response.get("choices")
        if choices:
            first = choices[0]
            message = getattr(first, "message", None)
            if message is None and isinstance(first, dict):
                message = first.get("message")
            reasoning = self._object_field(message, "reasoning_content", "")
            if isinstance(reasoning, str):
                self.last_reasoning_chars = len(reasoning)

        usage = getattr(response, "usage", None)
        if usage is None and isinstance(response, dict):
            usage = response.get("usage")
        if usage:
            self.last_completion_tokens = self._object_field(
                usage, "completion_tokens", None
            )
            details = self._object_field(usage, "completion_tokens_details", None)
            if details:
                self.last_reasoning_tokens = self._object_field(
                    details, "reasoning_tokens", None
                )

    @staticmethod
    def _object_field(obj, key, default=None):
        if obj is None:
            return default
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    @staticmethod
    def _failure_kind(err):
        msg = str(err).lower()
        if "truncated at max_tokens" in msg:
            return "output_truncated"
        if "context" in msg and ("length" in msg or "tokens" in msg):
            return "context_too_large"
        if "timed out" in msg or "timeout" in msg:
            return "timeout"
        if "authentication" in msg or "api key" in msg:
            return "authentication"
        if "quota" in msg or "usage limit" in msg:
            return "quota"
        if "empty" in msg or "unsupported response payload" in msg:
            return "empty_response"
        return "provider_error"

    @staticmethod
    def _is_temperature_error(exc):
        msg = str(exc).lower()
        return "temperature" in msg and ("only" in msg or "invalid" in msg)

    @property
    def effective_temperature(self):
        """The temperature actually sent, which the provider may have dictated."""
        return self.temperature if self._forced_temperature is None else self._forced_temperature

    @staticmethod
    def _finish_reason(response):
        choices = getattr(response, "choices", None)
        if not choices:
            return ""
        first = choices[0]
        fr = getattr(first, "finish_reason", None)
        if fr is None and isinstance(first, dict):
            fr = first.get("finish_reason")
        return fr if isinstance(fr, str) else ""

    def _record_provenance(self, response):
        served = self._response_field(response, "model")
        rid = self._response_field(response, "id")
        with self._prov_lock:
            if served:
                self.last_served_model = served
                self.served_models[served] += 1
            if rid:
                self.last_response_id = rid

    @staticmethod
    def _response_field(response, key):
        if response is None:
            return ""
        value = getattr(response, key, None)
        if value is None and isinstance(response, dict):
            value = response.get(key)
        return value.strip() if isinstance(value, str) else ""

    def _build_sampling_kwargs(self):
        # Greedy decoding (pass@1 protocol): only send temperature so providers
        # that reject temperature+top_p together do not error out.
        if self.temperature == 0 or "claude" in self.model.lower():
            return {"temperature": self.temperature}
        return {"temperature": self.temperature, "top_p": self.top_p}

    def _extract_error_message(self, response):
        if response is None:
            return ""
        err_obj = getattr(response, "error", None)
        if err_obj is None and hasattr(response, "model_dump"):
            dumped = response.model_dump()
            if isinstance(dumped, dict):
                err_obj = dumped.get("error")
        if isinstance(err_obj, dict):
            msg = err_obj.get("message")
            if isinstance(msg, str) and msg.strip():
                return msg.strip()
        return ""

    def _extract_text(self, response):
        if response is None:
            return ""

        choices = getattr(response, "choices", None)
        if choices:
            first = choices[0]
            message = getattr(first, "message", None)
            if message is None and isinstance(first, dict):
                message = first.get("message")

            if message is not None:
                content = getattr(message, "content", None)
                if content is None and isinstance(message, dict):
                    content = message.get("content")

                text = self._content_to_text(content)
                if text:
                    return text

            direct_text = getattr(first, "text", None)
            if direct_text is None and isinstance(first, dict):
                direct_text = first.get("text")
            if isinstance(direct_text, str) and direct_text.strip():
                return direct_text.strip()

        output_text = getattr(response, "output_text", None)
        if isinstance(output_text, str) and output_text.strip():
            return output_text.strip()

        if isinstance(response, dict):
            output_text = response.get("output_text")
            if isinstance(output_text, str) and output_text.strip():
                return output_text.strip()
            text = self._content_to_text(response.get("content"))
            if text:
                return text

        return ""


    def _content_to_text(self, content):
        if isinstance(content, str):
            return content.strip()

        if isinstance(content, list):
            chunks = []
            for item in content:
                if isinstance(item, str):
                    text = item.strip()
                    if text:
                        chunks.append(text)
                    continue

                text = None
                if isinstance(item, dict):
                    text = item.get("text") or item.get("content")
                else:
                    text = getattr(item, "text", None)
                    if text is None:
                        text = getattr(item, "content", None)

                if isinstance(text, str) and text.strip():
                    chunks.append(text.strip())

            return "\n".join(chunks).strip()

        return ""


    def _preview_response(self, response, max_len=800):
        if response is None:
            return "None"

        try:
            if hasattr(response, "model_dump"):
                dumped = response.model_dump()
            elif isinstance(response, dict):
                dumped = response
            else:
                dumped = str(response)
            text = str(dumped)
        except Exception:
            text = str(response)

        if len(text) > max_len:
            return text[:max_len] + "..."
        return text


    _QUOTA_SIGNALS = (
        "usagelimiterror", "monthly usage limit", "quota", "insufficient_quota",
        "billing", "exceeded your current",
    )

    @classmethod
    def _is_quota_error(cls, err):
        msg = str(err).lower()
        if any(s in msg for s in cls._QUOTA_SIGNALS):
            return True
        # A 429 that is not a rate limit but a spend cap: rate limits say so.
        return "429" in msg and "rate limit" not in msg

    def _should_retry(self, err):
        msg = str(err).lower()
        if self._is_quota_error(err):
            return False
        non_retriable_signals = [
            "invalid_request_error",
            "model_not_found",
            "context_length_exceeded",
            "unsupported",
            "no such model",
            "insufficient_quota",
            "authentication",
            "api key",
            "validationexception",
            "aws_invoke_error",
            "cannot both be specified",
            "truncated at max_tokens",
            "stream could not be cancelled",
            # A model the gateway will not serve over this wire format is not a
            # transient failure. Without this the loop retried it to exhaustion
            # -- 100s timeout plus a 2s sleep per attempt -- which is most of
            # why a pilot over four models took so long to tell us one of them
            # was unusable. Note the list already had "unsupported" (one word)
            # and the provider says "not supported" (two).
            "not supported",
            "not available",
        ]
        return not any(signal in msg for signal in non_retriable_signals)


    def run_with_timeout(self, func, timeout, on_timeout=None):
        """Outer bound that actually returns when it fires.

        Wrapping the call in `with ThreadPoolExecutor(...)` would not do: on
        timeout it raises, the `with` exits, and `shutdown(wait=True)` blocks
        until the very call that had just "timed out" finishes on its own -- so
        the timeout would only change which exception you eventually get, never
        when.

        The real idle-socket bound is on the HTTP client (`timeout=` below).
        This stays as a total-time backstop. For a stream it first closes the
        response and waits briefly for the reader to exit; an uncloseable
        reader is never retried concurrently, which prevents an old request
        from racing a new attempt and mutating shared provenance.
        """
        result = []
        error = []

        def invoke():
            try:
                result.append(func())
            except BaseException as exc:  # propagate the provider exception verbatim
                error.append(exc)

        # ThreadPoolExecutor workers are non-daemon.  ``shutdown(wait=False)``
        # therefore still kept the Python process alive after Ctrl-C until a
        # stuck request returned.  A daemon backstop has the intended semantics:
        # abandon a call that outlives the HTTP timeout without making the whole
        # sweep uninterruptible.
        worker = threading.Thread(target=invoke, daemon=True)
        worker.start()
        worker.join(timeout)
        if worker.is_alive():
            if on_timeout is not None:
                on_timeout()
                # A closed HTTP stream should release its reader promptly. Do
                # not begin a retry while an old stream can still mutate this
                # inference's metadata in the background.
                worker.join(5)
                if worker.is_alive():
                    raise TimeoutError(
                        f"call exceeded {timeout}s and stream could not be cancelled"
                    )
            raise TimeoutError(f"call exceeded {timeout}s")
        if error:
            raise error[0]
        return result[0] if result else None
