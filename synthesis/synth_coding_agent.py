import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
import select
import shlex
import subprocess
import tempfile
import time
import uuid
import re

from pathlib import Path
from typing import List
from typing import Optional

from context_synth import BENCHMARK_DIR
from synth_llm import build_prompt
from synth_llm import compose_full_dl
from synth_llm import extract_query
from synth_llm import load_or_build_context
from synth_llm import _feedback_from_result


AGENT_PROTOCOL_VERSION = 5


class AgentTurnError(RuntimeError):
    def __init__(self, message: str, command: List[str], *, failure_kind: str = "agent_error",
                 elapsed_seconds: Optional[float] = None):
        super().__init__(message)
        self.command = command
        self.failure_kind = failure_kind
        self.elapsed_seconds = elapsed_seconds


_QUOTA_MARKERS = (
    "balance insufficient",
    "insufficient balance",
    "insufficient credit",
    "credit balance",
    "insufficient_quota",
    "quota exceeded",
    "quota_exceeded",
    "余额不足",
)

_ACCOUNT_DISABLED_MARKERS = (
    "organization has been disabled",
    "organisation has been disabled",
    "organization is disabled",
    "organisation is disabled",
    "organization has been deactivated",
    "organisation has been deactivated",
    "account has been disabled",
    "account is disabled",
    "account has been deactivated",
    "organization_disabled",
    "organisation_disabled",
    "account_disabled",
)


def _is_quota_error(message: str) -> bool:
    """Recognize terminal billing/quota failures emitted by agent CLIs.

    Claude Code currently returns the provider's HTTP error as CLI text rather
    than a structured exception, so this classification must happen before the
    generic non-zero-exit path. HTTP 429 is deliberately not included: rate
    limiting is transient, whereas 402/balance exhaustion requires operator
    action and must stop the whole grid.
    """
    text = (message or "").lower()
    http_402 = bool(re.search(
        r"(?:api\s*error|http|status|error|code)[^\n]{0,48}\b402\b"
        r"|\b402\b[^\n]{0,48}(?:error|payment|required|balance)",
        text,
    ))
    return http_402 or any(
        marker in text for marker in _QUOTA_MARKERS
    )


def _is_account_disabled_error(message: str) -> bool:
    """Recognize non-retryable provider account/organization shutdowns.

    The Claude endpoint reports this state as HTTP 400, so status-code based
    classification alone treats it as an ordinary per-request CLI failure and
    lets the harness send every remaining case to an unusable account.
    """
    text = str(message or "").lower()
    return any(marker in text for marker in _ACCOUNT_DISABLED_MARKERS)


def _provider_fatal_kind(message: str) -> str:
    """Return the operator-action failure kind, or an empty string."""
    if _is_quota_error(message):
        return "quota"
    if _is_account_disabled_error(message):
        return "account_disabled"
    return ""


def _is_cli_usage_error(message: str) -> bool:
    """Recognize command-line parser failures, including legacy exit-0 logs."""
    text = str(message or "").lower()
    parser_error = any(marker in text for marker in (
        "unexpected argument", "unexpected option", "unknown argument",
        "unknown option", "unrecognized argument", "unrecognized option",
        "invalid value for",
    ))
    return parser_error and ("usage:" in text or "for more information" in text)


import agent_container
import measurement_v1


def _is_codex_default_mode(agent: str, agent_cmd: Optional[str]) -> bool:
    return agent_cmd is None and shlex.split(agent)[0] == "codex"


def _is_claude_default_mode(agent: str, agent_cmd: Optional[str]) -> bool:
    return agent_cmd is None and shlex.split(agent)[0] == "claude"


def _agent_cli_version(agent: str) -> str:
    """The CLI version string, captured at run time.

    Agent tooling changes faster than models do, so a run records which build
    produced it rather than asserting a version that may since have moved. Returns "" if the CLI does not
    answer, which is itself worth recording.
    """
    exe = shlex.split(agent)[0] if agent else ""
    if not exe:
        return ""
    for flag in ("--version", "-v", "version"):
        try:
            proc = subprocess.run([exe, flag], capture_output=True, text=True, timeout=20)
        except (OSError, subprocess.SubprocessError):
            continue
        out = (proc.stdout or proc.stderr or "").strip()
        if proc.returncode == 0 and out:
            return out.splitlines()[0].strip()[:120]
    return ""


_CLI_VERSION_CACHE = {}


def _agent_cli_version_cached(agent: str) -> str:
    """`_agent_cli_version` is a subprocess call; one per run is enough.

    In container mode the host's CLI is not the one that ran, so reporting its
    version would record something that never executed -- the same class of
    error the pinned image exists to remove. Read it out of the image instead.
    """
    head = shlex.split(agent)[0] if agent else ""
    if head not in _CLI_VERSION_CACHE:
        if _CONTAINER_MODE:
            prov = _image_provenance_cached()
            _CLI_VERSION_CACHE[head] = prov.get(f"{head}_version", "unknown")
        else:
            _CLI_VERSION_CACHE[head] = _agent_cli_version(agent)
    return _CLI_VERSION_CACHE[head]


_IMAGE_PROVENANCE: dict = {}


def _image_provenance_cached() -> dict:
    if not _IMAGE_PROVENANCE:
        _IMAGE_PROVENANCE.update(agent_container.image_provenance())
    return _IMAGE_PROVENANCE


def _resolve_default_model_for_agent(agent: str) -> str:
    return _model_inventory().resolve_default_model_for_agent(agent)


def _model_inventory():
    """The shared registry in evaluation/; imported lazily to keep this module
    importable from any working directory."""
    import sys
    from pathlib import Path as _P
    eval_dir = str(_P(__file__).resolve().parent.parent / "evaluation")
    if eval_dir not in sys.path:
        sys.path.insert(0, eval_dir)
    import model_inventory
    return model_inventory


def _model_to_path_tag(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model.strip())


def _method_tag(agent: str, method: str) -> str:
    return f"{agent}_{method}"


def _load_eval_tools():
    eval_dir = Path(__file__).resolve().parent.parent / "evaluation"

    common_eval_path = eval_dir / "common_eval.py"
    metrics_path = eval_dir / "metrics.py"

    common_eval_spec = importlib.util.spec_from_file_location("common_eval_dynamic", common_eval_path)
    if common_eval_spec is None or common_eval_spec.loader is None:
        raise ImportError(f"Cannot load module from {common_eval_path}")
    common_eval_module = importlib.util.module_from_spec(common_eval_spec)
    common_eval_spec.loader.exec_module(common_eval_module)

    metrics_spec = importlib.util.spec_from_file_location("metrics_dynamic", metrics_path)
    if metrics_spec is None or metrics_spec.loader is None:
        raise ImportError(f"Cannot load module from {metrics_path}")
    metrics_module = importlib.util.module_from_spec(metrics_spec)
    metrics_spec.loader.exec_module(metrics_module)

    return (
        common_eval_module.evaluate_query_file_for_case,
        common_eval_module.case_demo_dirs,
        metrics_module.calculate_from_confusion,
    )


def _build_synthesis_prompt(base_prompt: str) -> str:
    return (
        "You are in a persistent coding-agent session for one Datalog case.\n"
        "Complete this workflow autonomously inside this session: query synthesis -> validation -> iterative refinement -> final query.\n"
        "Do not output comments in Datalog.\n"
        "Return exactly one ```datalog``` code block in your final answer for this turn.\n\n"
        "Task Context:\n"
        + base_prompt
    )


def _build_continue_prompt(feedback: str) -> str:
    return (
        "Continue refining within this same session memory.\n"
        "Your previous program was executed on the development I/O instance. Feedback:\n"
        + feedback
        + "\nCorrect the program using this feedback.\n"
        "Return exactly one ```datalog``` code block and no comments.\n"
    )


# Container isolation. Set once from main(); the three launch paths below read
# it rather than each deciding for itself, so there is one place where the
# boundary is either on or off.
_CONTAINER_MODE = True


def _require_container() -> None:
    """Fail loudly rather than run the agent under the isolation we replaced."""
    if not _CONTAINER_MODE:
        return
    why = agent_container.available()
    if why:
        raise agent_container.ContainerUnavailable(
            f"{why}\n"
            "Refusing to fall back to the host: results would be labelled as\n"
            "container runs while carrying the weaker isolation. Fix the setup\n"
            "(docker/README.md) or pass --no-container and say so in the record."
        )


def _launch(cmd: List[str], workdir: Path,
            session_agent: Optional[str] = None) -> tuple:
    """(argv, cwd) for a launch. In a container the cwd comes from `-w /work`."""
    if not _CONTAINER_MODE:
        return cmd, str(workdir)
    return agent_container.build_command(
        cmd, workdir, session_agent=session_agent,
    ), None


def _run_agent_command(
    cmd: List[str], workdir: Path, timeout_sec: int, input_text: Optional[str] = None,
    session_agent: Optional[str] = None,
) -> subprocess.CompletedProcess:
    """Run one bounded CLI turn and remove its container on interruption.

    Killing an attached ``docker run --rm`` client does not stop the container.
    Without an explicit cidfile cleanup, every timed-out model call leaves its
    Codex/Claude process running indefinitely and retries multiply the leak.
    """
    if _CONTAINER_MODE and session_agent:
        session_dir = workdir / f".{session_agent}-sessions"
        session_dir.mkdir(mode=0o700, exist_ok=True)
        # The image entrypoint first writes credentials to the tmpfs HOME.  Only
        # after that do we point only the CLI's session subdirectory at the
        # per-case mount. Mounting the whole config would persist the key.
        if session_agent == "claude":
            session_path = '$HOME/.claude/projects'
        elif session_agent == "codex":
            session_path = '$HOME/.codex/sessions'
        else:
            raise ValueError(f"unsupported session agent: {session_agent}")
        cmd = [
            "sh", "-c",
            f'rm -rf "{session_path}" && mkdir -p "$(dirname "{session_path}")" && '
            f'ln -s /sessions "{session_path}" && exec "$@"',
            f"{session_agent}-session-wrapper",
            *cmd,
        ]
    argv, cwd = _launch(cmd, workdir, session_agent=session_agent)
    cidfile = None
    if _CONTAINER_MODE:
        cidfile = workdir / f".dlb-container-{uuid.uuid4().hex}.cid"
        # argv starts with ``docker run``.  --cidfile is a daemon-side path, so
        # use the host path rather than the container's /work spelling.
        argv[2:2] = ["--cidfile", str(cidfile)]
        if input_text is None:
            # ``docker run -i`` deliberately keeps the container's stdin open.
            # That makes CLIs which also inspect stdin wait forever even when
            # their prompt was supplied as an argv value.
            argv.remove("-i")

    try:
        return subprocess.run(
            argv,
            input=input_text,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout_sec,
            check=False,
            cwd=cwd,
        )
    except BaseException:
        if cidfile is not None:
            # The daemon writes the cidfile before starting the container.  A
            # short retry also covers interruption during container creation.
            cid = ""
            for _ in range(20):
                try:
                    cid = cidfile.read_text(encoding="utf-8").strip()
                except OSError:
                    pass
                if cid:
                    break
                time.sleep(0.05)
            if cid:
                subprocess.run(
                    ["docker", "rm", "-f", cid],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=20,
                    check=False,
                )
        raise
    finally:
        if cidfile is not None:
            cidfile.unlink(missing_ok=True)


def _render_agent_launch_command(
    agent: str,
    agent_cmd: Optional[str],
) -> List[str]:
    if agent_cmd:
        if "{prompt}" in agent_cmd or "{prompt_file}" in agent_cmd:
            raise ValueError("Persistent session mode does not support {prompt} or {prompt_file} in --agent_cmd")
        template = agent_cmd
        rendered = template.format(agent=agent)
        return shlex.split(rendered)

    # Default behavior: launch agent CLI command directly.
    return shlex.split(agent)


def _set_nonblocking(fd: int) -> None:
    flags = fcntl.fcntl(fd, fcntl.F_GETFL)
    fcntl.fcntl(fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)


def _start_agent_session(
    agent: str,
    workdir: Path,
    agent_cmd: Optional[str] = None,
) -> tuple:
    cmd = _render_agent_launch_command(agent=agent, agent_cmd=agent_cmd)
    argv, cwd = _launch(cmd, workdir)
    proc = subprocess.Popen(
        argv,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=False,
        bufsize=0,
        cwd=cwd,
    )

    if proc.stdout is None or proc.stderr is None or proc.stdin is None:
        raise RuntimeError("Failed to start persistent agent session")

    _set_nonblocking(proc.stdout.fileno())
    _set_nonblocking(proc.stderr.fileno())
    return proc, cmd


def _stop_agent_session(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=3)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=3)


def _collect_session_output(proc: subprocess.Popen, timeout_sec: int, idle_sec: float = 1.5) -> tuple:
    if proc.stdout is None or proc.stderr is None:
        return "", ""

    out_parts = []
    err_parts = []
    stdout_fd = proc.stdout.fileno()
    stderr_fd = proc.stderr.fileno()
    fds = [stdout_fd, stderr_fd]

    start = time.time()
    last_data = time.time()

    while True:
        now = time.time()
        if now - start > timeout_sec:
            break
        if (out_parts or err_parts) and (now - last_data > idle_sec):
            break

        ready, _, _ = select.select(fds, [], [], 0.2)
        if not ready:
            continue

        for fd in ready:
            try:
                chunk = os.read(fd, 4096)
            except BlockingIOError:
                continue

            if not chunk:
                continue

            text = chunk.decode("utf-8", errors="replace")
            last_data = time.time()
            if fd == stdout_fd:
                out_parts.append(text)
            else:
                err_parts.append(text)

    return "".join(out_parts).strip(), "".join(err_parts).strip()


def _session_send_and_receive(
    proc: subprocess.Popen,
    prompt: str,
    timeout_sec: int,
) -> dict:
    if proc.stdin is None:
        raise RuntimeError("Persistent session stdin unavailable")

    if proc.poll() is not None:
        raise RuntimeError(f"Agent session exited early with code {proc.returncode}")

    proc.stdin.write((prompt + "\n").encode("utf-8"))
    proc.stdin.flush()

    output, stderr = _collect_session_output(proc=proc, timeout_sec=timeout_sec)
    if not output and stderr:
        output = stderr
    if not output:
        raise RuntimeError("Agent session returned empty output")

    return {
        "prompt": prompt,
        "response": output,
        "stderr": stderr,
    }


def _parse_codex_jsonl(stdout_text: str) -> dict:
    thread_id = None
    fallback_response = ""
    events = []

    for raw in stdout_text.splitlines():
        line = raw.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue

        events.append(event)

        if event.get("type") == "thread.started":
            thread_id = event.get("thread_id")

        if event.get("type") == "item.completed":
            item = event.get("item") or {}
            text = item.get("text")
            if isinstance(text, str) and text.strip():
                fallback_response = text.strip()

    return {
        "thread_id": thread_id,
        "response": fallback_response,
        "events": events,
    }


def _run_codex_exec_turn(
    prompt: str,
    timeout_sec: int,
    model: str,
    workdir: Path,
    thread_id: Optional[str] = None,
) -> dict:
    # The file has to live inside the working directory: it is the only path
    # both sides can see, since the container mounts nothing else. The name codex
    # is given is therefore the container-side one, while the harness reads back
    # the host-side path the mount maps to.
    with tempfile.NamedTemporaryFile(mode="w", suffix=".codex-last.txt", delete=False,
                                     dir=str(workdir), encoding="utf-8") as tf:
        output_file = Path(tf.name)
    arg_output_file = ("/work/" + output_file.name) if _CONTAINER_MODE else str(output_file)

    try:
        if thread_id:
            # `codex exec resume` does not expose the top-level `--sandbox`
            # option. Its equivalent config override is accepted by resume.
            isolation = _model_inventory().agent_isolation("codex")["args"]
            if isolation[:1] != ["--sandbox"] or len(isolation) != 2:
                raise RuntimeError(f"unsupported Codex isolation args: {isolation!r}")
            resume_isolation = ["--config", f'sandbox_mode="{isolation[1]}"']
            cmd = [
                "codex",
                "exec",
                "resume",
                "--skip-git-repo-check",
                "--model",
                model,
                "--json",
                "--output-last-message",
                arg_output_file,
                *resume_isolation,
                thread_id,
                prompt,
            ]
        else:
            cmd = [
                "codex",
                "exec",
                "--skip-git-repo-check",
                "--model",
                model,
                "--json",
                "--output-last-message",
                arg_output_file,
                *_model_inventory().agent_tool_policy("codex")["args"],
                *_model_inventory().agent_isolation("codex")["args"],
                prompt,
            ]

        # Codex enables web search only when --search is passed. That makes the
        # restriction an absence, which is easy to reintroduce by accident, so
        # assert it instead of trusting it.
        assert "--search" not in cmd, "codex web search must stay disabled (see AGENT_TOOL_POLICY)"
        # Resume has no --sandbox flag, so it receives the same policy through
        # a config override. The container is the primary experiment boundary.
        if thread_id:
            assert any(arg.startswith('sandbox_mode="workspace-write"') for arg in cmd)
            assert "--sandbox" not in cmd
        else:
            assert "--sandbox" in cmd, "codex must run sandboxed (see AGENT_ISOLATION)"
        assert "--dangerously-bypass-approvals-and-sandbox" not in cmd

        last_err = ""
        for attempt in range(2):
            try:
                proc = _run_agent_command(
                    cmd=cmd, workdir=workdir, timeout_sec=timeout_sec,
                    session_agent="codex",
                )
            except subprocess.TimeoutExpired:
                last_err = f"codex exec timed out after {timeout_sec}s"
                if attempt == 0:
                    continue
                raise AgentTurnError(last_err, cmd)

            if proc.returncode != 0:
                error_text = (proc.stderr or proc.stdout or "codex exec failed").strip()
                raise AgentTurnError(
                    error_text, cmd,
                    failure_kind=_provider_fatal_kind(error_text) or "cli_error",
                )

            parsed = _parse_codex_jsonl(proc.stdout or "")
            next_thread_id = parsed.get("thread_id") or thread_id

            response = ""
            if output_file.exists():
                response = output_file.read_text(encoding="utf-8").strip()
            if not response:
                response = parsed.get("response", "")

            combined_output = "\n".join((
                response, str(proc.stderr or ""), str(proc.stdout or ""),
            ))
            provider_fatal = _provider_fatal_kind(combined_output)
            if provider_fatal or _is_cli_usage_error(combined_output):
                raise AgentTurnError(
                    combined_output.strip(), cmd,
                    failure_kind=provider_fatal or "cli_usage_error",
                )

            if response:
                return {
                    "prompt": prompt,
                    "response": response,
                    "stderr": (proc.stderr or "").strip(),
                    "command": cmd,
                    "thread_id": next_thread_id,
                    # Preserve tool/lifecycle events for the egress audit; the
                    # former dialogue stored only the final message text.
                    "event_log": parsed.get("events", []),
                }

            last_err = "codex exec returned empty response"
            if attempt == 0:
                continue
            raise AgentTurnError(last_err, cmd)

        raise AgentTurnError(last_err or "codex exec failed", cmd)
    finally:
        try:
            output_file.unlink(missing_ok=True)
        except OSError:
            pass


def _claude_print_command(prompt: str, session_id: str, model: str,
                          resume_session: bool) -> List[str]:
    session_args = ["--resume", session_id] if resume_session else ["--session-id", session_id]
    return [
        "claude",
        "-p",
        # --allowedTools accepts a variable-length list.  A positional prompt
        # placed after it is consumed as another tool name, leaving --print
        # with no input.  Bind the optional -p argument before any options.
        prompt,
        "--model",
        model,
        "--output-format",
        "text",
        *session_args,
        # Allowlist over the built-in tool set: no web tool is listed, so the
        # agent cannot look up the reference program (see AGENT_TOOL_POLICY).
        *_model_inventory().agent_tool_policy("claude")["args"],
    ]


def _run_claude_print_turn(
    prompt: str,
    timeout_sec: int,
    session_id: str,
    model: str,
    workdir: Path,
    resume_session: bool = False,
) -> dict:
    cmd = _claude_print_command(prompt, session_id, model, resume_session)

    started = time.monotonic()
    try:
        proc = _run_agent_command(
            cmd=cmd, workdir=workdir, timeout_sec=timeout_sec,
            session_agent="claude",
        )
    except subprocess.TimeoutExpired:
        # Reissuing the same logical turn can duplicate a server-side request and
        # used to turn one 600-second timeout into 20 minutes.  The timed-out
        # container is removed by _run_agent_command; stop this case and retain
        # its best completed candidate instead of replaying the turn.
        raise AgentTurnError(
            f"claude print timed out after {timeout_sec}s", cmd,
            failure_kind="timeout",
            elapsed_seconds=round(time.monotonic() - started, 3),
        )

    elapsed = round(time.monotonic() - started, 3)
    if proc.returncode != 0:
        error_text = "\n".join(
            part.strip() for part in (proc.stderr or "", proc.stdout or "")
            if part.strip()
        ) or "claude print failed"
        raise AgentTurnError(
            error_text, cmd,
            failure_kind=_provider_fatal_kind(error_text) or "cli_error",
            elapsed_seconds=elapsed,
        )

    response = (proc.stdout or "").strip()
    if not response:
        response = (proc.stderr or "").strip()
    if not response:
        raise AgentTurnError(
            "claude print returned empty response", cmd,
            failure_kind="empty_response", elapsed_seconds=elapsed,
        )
    combined_output = "\n".join((response, (proc.stderr or "").strip()))
    provider_fatal = _provider_fatal_kind(combined_output)
    if provider_fatal:
        raise AgentTurnError(
            combined_output, cmd, failure_kind=provider_fatal, elapsed_seconds=elapsed,
        )
    if _is_cli_usage_error(combined_output):
        raise AgentTurnError(
            combined_output, cmd, failure_kind="cli_usage_error",
            elapsed_seconds=elapsed,
        )
    return {
        "prompt": prompt,
        "response": response,
        "stderr": (proc.stderr or "").strip(),
        "command": cmd,
        "session_id": session_id,
        "session_action": "resume" if resume_session else "start",
        "elapsed_seconds": elapsed,
        "attempts": 1,
    }


def _evaluate_query(case_record: dict, full_dl: str, evaluate_query_file_for_case,
                    case_demo_dirs, calculate_from_confusion):
    with tempfile.TemporaryDirectory(prefix=f"coding_agent_{case_record['id']}_") as tmp:
        tmp_path = Path(tmp)
        query_file = tmp_path / f"{case_record['id']}.dl"
        query_file.write_text(full_dl, encoding="utf-8")
        result = evaluate_query_file_for_case(
            case_record,
            query_file,
            include_counterexamples=True,
            variant_dirs=case_demo_dirs(case_record["id"]),
            evaluation_pool="demo",
        )

    metric = calculate_from_confusion(result["tp"], result["fp"], result["fn"])
    # Same rule as common_eval.is_exact, which this module does not import:
    # an unfinished (timed-out) program is never exact.
    perfect_match = bool(result["compile_ok"] and not result.get("timed_out")
                         and result["fp"] == 0 and result["fn"] == 0)
    return result, metric, perfect_match


def _atomic_json(path: Path, payload) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _prompt_sha256(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def _best_round(trace):
    return max(trace, key=lambda item: (
        bool(item.get("perfect_match")),
        float(item.get("f1", -1.0)),
        bool(item.get("compile_ok")),
        -(int(item.get("fp", 0)) + int(item.get("fn", 0))),
    ))


def _checkpoint_validity(checkpoint: dict, trace: list, dialogue: list) -> tuple:
    """Return (valid, error, failure_kind) for old and current checkpoints.

    Protocol-v2 rows predate explicit validity fields. They are accepted only
    when their trace contains no recorded failed turn and no quota marker. This
    preserves verified successful work while making every known 402/timeout/
    empty/CLI-error row non-reusable. Protocol-v3 rows must opt in explicitly.
    """
    texts = [str(checkpoint.get("inference_error") or ""),
             str(checkpoint.get("response") or "")]
    for item in trace:
        if isinstance(item, dict):
            texts.extend((str(item.get("response") or ""),
                          str(item.get("failure_kind") or "")))
    for item in dialogue:
        if isinstance(item, dict):
            texts.extend((str(item.get("response") or ""),
                          str(item.get("stderr") or "")))
    provider_fatal = next(
        (_provider_fatal_kind(text) for text in texts if _provider_fatal_kind(text)),
        "",
    )
    quota = provider_fatal == "quota"
    cli_usage = any(_is_cli_usage_error(text) for text in texts)
    failed_turns = [
        item for item in trace
        if isinstance(item, dict) and item.get("inference_ok") is False
    ]
    last_failure = failed_turns[-1] if failed_turns else {}
    error = str(last_failure.get("response") or checkpoint.get("inference_error") or "")
    kind = provider_fatal or ("cli_usage_error" if cli_usage else str(
        last_failure.get("failure_kind") or checkpoint.get("failure_kind") or ""
    ))

    if checkpoint.get("agent_protocol_version") == AGENT_PROTOCOL_VERSION:
        valid = (
            checkpoint.get("checkpoint_valid") is True
            and checkpoint.get("inference_ok") is True
            and checkpoint.get("terminal_failure") is False
            and not checkpoint.get("inference_error")
            and not failed_turns
            and not provider_fatal
            and not cli_usage
        )
        return valid, error, kind

    # Compatibility classification for older explicit checkpoints. Their
    # callers decide whether the old protocol is scientifically reusable.
    valid = (
        checkpoint.get("agent_protocol_version") in {2, 3}
        and bool((checkpoint.get("response") or "").strip())
        and bool(trace)
        and not failed_turns
        and not provider_fatal
        and not cli_usage
    )
    return valid, error, kind


def _agent_row(
    *, cid, run_id, best, trace, resolved_model, agent, method, temperature,
    prompt_digest, max_iterations, timeout_sec, agent_cmd, checkpoint_origin,
):
    if _is_claude_default_mode(agent=agent, agent_cmd=agent_cmd):
        session_persistence = "per-case-session-mount+claude-resume"
        timeout_retries = 0
    elif _is_codex_default_mode(agent=agent, agent_cmd=agent_cmd):
        session_persistence = "per-case-session-mount+codex-thread-resume"
        timeout_retries = 1
    else:
        session_persistence = "single-persistent-process"
        timeout_retries = 0
    failed_turns = [item for item in trace if item.get("inference_ok") is False]
    terminal = failed_turns[-1] if failed_turns else None
    checkpoint_valid = bool(trace) and terminal is None
    return {
        "id": cid,
        "run": run_id,
        "response": best.get("response", ""),
        "query": best.get("query", ""),
        "agent": agent,
        "agent_cli_version": _agent_cli_version_cached(agent),
        "tools_enforced": _model_inventory().agent_tool_policy(agent)["enforced"],
        "tools_note": _model_inventory().agent_tool_policy(agent)["note"],
        "isolation": "container" if _CONTAINER_MODE else "host",
        "image_id": _image_provenance_cached().get("image_id") if _CONTAINER_MODE else None,
        "souffle_version": _image_provenance_cached().get("souffle_version") if _CONTAINER_MODE else None,
        "model": resolved_model,
        "endpoint": agent_container.agent_endpoints().get(agent),
        "credential_env": agent_container.AGENT_API_KEY_ENV,
        "resume_source": os.environ.get("AGENT_RESUME_SOURCE", "protocol-v5-run"),
        "method": method,
        "selection_pool": "demo",
        "scoring_pool": "eval",
        # The two fields above say where selection happens and where the
        # official score will be computed; this one says which pool the numbers
        # *in this record* came from, and the answer is demo -- `_evaluate_query`
        # is the only scoring call in this file and it is pinned to the demo
        # pool by protocol v5.  Without this field the record showed
        # scoring_pool=eval next to a perfect_match computed on demo, and the
        # pair was read as an exact-match rate more than once.  The gap is not
        # small: 130 demo-perfect vs 109 eval-exact for Codex, 123 vs 110 for
        # Claude.  Only eval_coding_agent.py produces exact match.
        "metrics_pool": "demo",
        "temperature": temperature,
        "iterations_used": best.get("iteration", 0),
        "compile_ok": bool(best.get("compile_ok")),
        "perfect_match": bool(best.get("perfect_match")),
        "f1": best.get("f1", 0.0),
        "tp": best.get("tp", 0),
        "fp": best.get("fp", 0),
        "fn": best.get("fn", 0),
        "feedback": best.get("feedback", ""),
        "best_iteration": best.get("iteration", 0),
        "session_turns": len(trace),
        "dialogue_file": f"{cid}.dialogue.json",
        "agent_protocol_version": AGENT_PROTOCOL_VERSION,
        "session_persistence": session_persistence,
        "timeout_retries": timeout_retries,
        "prompt_sha256": prompt_digest,
        "max_iterations": max_iterations,
        "timeout_sec": timeout_sec,
        "agent_cmd": agent_cmd,
        "checkpoint_origin": checkpoint_origin,
        "checkpoint_valid": checkpoint_valid,
        "inference_ok": checkpoint_valid,
        "inference_error": (terminal or {}).get("response", ""),
        "failure_kind": (terminal or {}).get("failure_kind", ""),
        "terminal_failure": terminal is not None,
        "attempts": sum(item.get("attempts", 0) for item in trace),
    }


def _load_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _resume_row(
    *, run_dir, task, run_id, resolved_model, agent, method, temperature,
    prompt_digest, max_iterations, timeout_sec, agent_cmd,
):
    """Return a verified case row, or None when the case must be regenerated.

    New checkpoints are reusable only under an exact protocol/configuration
    match. Older explicit checkpoints are migrated only when that identity
    still matches and the first turn was already exact; default-agent legacy
    rows lack endpoint/image identity and are always rerun. Failed turns
    (including historical 402/CLI-usage rows) are never reused.
    """
    cid = task["case_id"]
    query_path = run_dir / f"{cid}.dl"
    trace_path = run_dir / f"{cid}.trace.json"
    dialogue_path = run_dir / f"{cid}.dialogue.json"
    checkpoint_path = run_dir / f"{cid}.agent.json"

    # A case the finalizer has already closed as a terminal zero keeps its row
    # and is not attempted again.  Without this, `--resume` would see
    # checkpoint_valid=false and reopen a retry budget that measurement-v1 has
    # already declared spent -- and a third attempt on a case selected precisely
    # because two attempts failed is success-conditioned retry, which biases the
    # number upward every time it happens to work.
    finalized = _load_json(checkpoint_path)
    if isinstance(finalized, dict) and measurement_v1.is_terminal_zero(finalized):
        return dict(finalized, checkpoint_origin="measurement-v1-terminal-zero")

    trace = _load_json(trace_path)
    dialogue = _load_json(dialogue_path)
    if (
        not query_path.is_file()
        or not isinstance(trace, list) or not trace
        or not isinstance(dialogue, list) or len(dialogue) != len(trace)
    ):
        return None

    checkpoint = _load_json(checkpoint_path)
    if isinstance(checkpoint, dict):
        expected = {
            "id": cid,
            "run": run_id,
            "agent": agent,
            "model": resolved_model,
            "method": method,
            "selection_pool": "demo",
            "scoring_pool": "eval",
            "temperature": temperature,
            "prompt_sha256": prompt_digest,
            "max_iterations": max_iterations,
            "timeout_sec": timeout_sec,
            "agent_cmd": agent_cmd,
            "isolation": "container" if _CONTAINER_MODE else "host",
            "agent_cli_version": _agent_cli_version_cached(agent),
            "image_id": (_image_provenance_cached().get("image_id")
                         if _CONTAINER_MODE else None),
        }
        if _is_claude_default_mode(agent=agent, agent_cmd=agent_cmd):
            expected.update({
                "session_persistence": "per-case-session-mount+claude-resume",
                "timeout_retries": 0,
            })
        elif _is_codex_default_mode(agent=agent, agent_cmd=agent_cmd):
            expected.update({
                "session_persistence": "per-case-session-mount+codex-thread-resume",
                "timeout_retries": 1,
            })
        if any(checkpoint.get(key) != value for key, value in expected.items()):
            return None
        checkpoint_protocol = checkpoint.get("agent_protocol_version")
        if checkpoint_protocol != AGENT_PROTOCOL_VERSION:
            return None
        checkpoint_valid, _, _ = _checkpoint_validity(checkpoint, trace, dialogue)
        if not checkpoint_valid:
            return None
        best = checkpoint
        origin = "protocol-v5-checkpoint"
    else:
        legacy_texts = []
        for item in trace + dialogue:
            if isinstance(item, dict):
                legacy_texts.extend((item.get("response"), item.get("stderr")))
        if (
            any(item.get("inference_ok") is False
                for item in trace if isinstance(item, dict))
            or any(_provider_fatal_kind(text) for text in legacy_texts)
            or any(_is_cli_usage_error(text) for text in legacy_texts)
        ):
            return None
        # Legacy default-agent artifacts have no endpoint/image identity.
        # Accepting even a first-turn exact legacy result would silently mix
        # providers, irrespective of the endpoint selected for the current run.
        if (_is_claude_default_mode(agent=agent, agent_cmd=agent_cmd)
                or _is_codex_default_mode(agent=agent, agent_cmd=agent_cmd)):
            return None
        best = _best_round(trace)
        origin = "legacy-verified"

    expected_dl = compose_full_dl(
        task["schema_def"], best.get("query", ""), case_id=cid,
    )
    try:
        if query_path.read_text(encoding="utf-8") != expected_dl:
            return None
    except OSError:
        return None

    if isinstance(checkpoint, dict):
        row = dict(checkpoint)
        # Protocol-v5 checkpoints created before endpoint provenance was added
        # came from the pinned zzz image. Preserve that historical endpoint
        # rather than falsely labelling a reused row with the current override.
        row.setdefault("endpoint", agent_container.DEFAULT_AGENT_BASE_URLS.get(agent))
        row.setdefault("credential_env", "EVAL_API_KEY")
        row.setdefault("resume_source", "pre-endpoint-provenance-checkpoint")
        row.update({
            "agent_protocol_version": AGENT_PROTOCOL_VERSION,
            "checkpoint_origin": origin,
            "checkpoint_valid": True,
            "inference_ok": True,
            "inference_error": "",
            "failure_kind": "",
            "terminal_failure": False,
            "resumed": True,
        })
        _atomic_json(checkpoint_path, row)
        return row

    row = _agent_row(
        cid=cid,
        run_id=run_id,
        best=best,
        trace=trace,
        resolved_model=resolved_model,
        agent=agent,
        method=method,
        temperature=temperature,
        prompt_digest=prompt_digest,
        max_iterations=max_iterations,
        timeout_sec=timeout_sec,
        agent_cmd=agent_cmd,
        checkpoint_origin=origin,
    )
    row["resumed"] = True
    _atomic_json(checkpoint_path, row)
    return row


def synthesize_with_coding_agent(
    dataset: str,
    agent: str,
    model: Optional[str] = None,
    method: str = "signature",
    case_id: str = None,
    max_iterations: int = 4,
    timeout_sec: int = 180,
    agent_cmd: Optional[str] = None,
    temperature: float = 0.0,
    num_runs: int = 1,
    resume: bool = False,
):
    if method not in {"signature", "description"}:
        raise ValueError("--method must be 'signature' or 'description'")
    if max_iterations <= 0:
        raise ValueError("--max_iterations must be > 0")
    if timeout_sec <= 0:
        raise ValueError("--timeout_sec must be > 0")
    if num_runs <= 0:
        raise ValueError("--num_runs must be > 0")

    _model_inventory().check_agent_supported(agent)
    resolved_model = model if model else _resolve_default_model_for_agent(agent)
    model_tag = _model_to_path_tag(resolved_model)

    # State the tool policy up front. An agent whose tools cannot be constrained
    # is the one fact a reader of these numbers most needs, and it must not be
    # discoverable only by reading the code afterwards.
    policy = _model_inventory().agent_tool_policy(agent)
    version = _agent_cli_version_cached(agent)
    print(f"[AGENT] {agent} | CLI version: {version or 'not reported'}")
    if policy["enforced"]:
        print(f"[TOOLS] constrained: {policy['note']}")
        if policy["args"]:
            print(f"[TOOLS] flags: {' '.join(policy['args'])}")
    else:
        print(f"[TOOLS] *** NOT CONSTRAINED *** {policy['note']}")
        print("[TOOLS] This run's agent may reach the network. The upstream artifacts for "
              "these tasks are public together with their reference programs, so retrieval "
              "is a leak channel, not merely extra capability -- report this agent's "
              "numbers with that caveat.")

    # Isolation is a separate axis from the tool allowlist and is stated
    # separately, because closing the web tools does not close an allowlisted
    # shell, and neither closes a working directory that contains the answer.
    iso = _model_inventory().agent_isolation(agent)
    if _CONTAINER_MODE:
        prov = _image_provenance_cached()
        print(f"[ISOLATION] container {prov['image']} ({prov['image_id']}) -- same "
              "boundary for every agent, so a gap between rows is a gap in the model")
        print("[ISOLATION] filesystem: only the per-case scratch dir is mounted; the "
              "repository is absent, so a reference program has no path to read")
        print("[ISOLATION] network: internal, no route off the host except a CONNECT "
              "allowlist holding the model endpoint alone")
    else:
        print("[ISOLATION] cwd: scratch directory per case (the repository, and with it "
              "benchmark/query/<case>.dl, is not visible)")
        if iso["network_blocked"]:
            print(f"[ISOLATION] network: blocked -- {iso['note']}")
        else:
            print(f"[ISOLATION] network: *** NOT BLOCKED *** {iso['note']}")

    evaluate_query_file_for_case, case_demo_dirs, calculate_from_confusion = _load_eval_tools()
    tasks = load_or_build_context(dataset=dataset, case_id=case_id)

    dataset_scope = dataset if dataset else "all"
    method_tag = _method_tag(agent=agent, method=method)

    if case_id:
        output_base_dir = BENCHMARK_DIR / "infer_data" / model_tag / method_tag / case_id
    else:
        output_base_dir = BENCHMARK_DIR / "infer_data" / model_tag / method_tag / dataset_scope

    incomplete_runs = 0
    for run_id in range(num_runs):
        run_dir = output_base_dir / f"run_{run_id}"
        run_dir.mkdir(parents=True, exist_ok=True)
        outcome = _synthesize_one_run(
            run_id=run_id,
            run_dir=run_dir,
            tasks=tasks,
            resolved_model=resolved_model,
            agent=agent,
            method=method,
            case_id=case_id,
            dataset_scope=dataset_scope,
            max_iterations=max_iterations,
            timeout_sec=timeout_sec,
            agent_cmd=agent_cmd,
            temperature=temperature,
            evaluate_query_file_for_case=evaluate_query_file_for_case,
            case_demo_dirs=case_demo_dirs,
            calculate_from_confusion=calculate_from_confusion,
            resume=resume,
        )
        if not outcome["complete"]:
            incomplete_runs += 1
        if outcome["quota_exhausted"]:
            print("[ABORT][QUOTA] Billing/quota exhaustion is not retryable; "
                  "stopping all remaining runs and grid cells.")
            return 3
        if outcome.get("fatal_error_kind"):
            print(f"[ABORT][PROVIDER] {outcome['fatal_error_kind']} is not retryable; "
                  "stopping all remaining runs and grid cells.")
            return 4

    print(f"[DONE] {num_runs} run(s) -> {output_base_dir}")
    return 1 if incomplete_runs else 0


def _synthesize_one_run(
    run_id,
    run_dir,
    tasks,
    resolved_model,
    agent,
    method,
    case_id,
    dataset_scope,
    max_iterations,
    timeout_sec,
    agent_cmd,
    temperature,
    evaluate_query_file_for_case,
    case_demo_dirs,
    calculate_from_confusion,
    resume,
):
    rows = []
    failures = []
    fatal_stop_kind = ""

    for task in tasks:
        cid = task["case_id"]
        case_record = {
            "id": cid,
            "output_relation": {x["signature"]: x.get("description", "") for x in task["schema_def"]["output"]},
        }

        base_prompt = build_prompt(task, method=method, case_examples=[])
        synthesis_prompt = _build_synthesis_prompt(base_prompt)
        prompt_digest = _prompt_sha256(synthesis_prompt)

        if resume:
            resumed_row = _resume_row(
                run_dir=run_dir,
                task=task,
                run_id=run_id,
                resolved_model=resolved_model,
                agent=agent,
                method=method,
                temperature=temperature,
                prompt_digest=prompt_digest,
                max_iterations=max_iterations,
                timeout_sec=timeout_sec,
                agent_cmd=agent_cmd,
            )
            if resumed_row is not None:
                rows.append(resumed_row)
                print(f"[RESUME][AGENT:{agent}][run {run_id}] {cid}: "
                      f"{resumed_row['checkpoint_origin']}")
                continue

        best = None
        trace = []
        dialogue_turns = []
        # The agent's own working directory. It must not be the repository root:
        # `benchmark/query/<case>.dl` is this task's reference program, and Read
        # and Bash are both allowlisted.
        agent_workdir_ctx = tempfile.TemporaryDirectory(prefix=f"agent_ws_{cid}_")
        agent_workdir = Path(agent_workdir_ctx.name)
        session_proc = None
        launch_cmd = None
        session_unhealthy = False
        codex_thread_id = None
        claude_session_id = None
        use_codex_exec = _is_codex_default_mode(agent=agent, agent_cmd=agent_cmd)
        use_claude_print = _is_claude_default_mode(agent=agent, agent_cmd=agent_cmd)
        use_persistent_session = not (use_codex_exec or use_claude_print)
        feedback = ""
        try:
            if use_claude_print:
                claude_session_id = str(uuid.uuid4())
            if use_persistent_session:
                session_proc, launch_cmd = _start_agent_session(
                    agent=agent, workdir=agent_workdir, agent_cmd=agent_cmd)

            for iteration in range(1, max_iterations + 1):
                if iteration == 1:
                    prompt = synthesis_prompt
                else:
                    prompt = _build_continue_prompt(feedback)
                command_for_turn = launch_cmd
                turn_elapsed = None
                turn_attempts = 1
                inference_ok = True
                failure_kind = ""

                try:
                    if use_codex_exec:
                        command_for_turn = ["codex", "exec"]
                        call_result = _run_codex_exec_turn(
                            prompt=prompt,
                            timeout_sec=timeout_sec,
                            model=resolved_model,
                            workdir=agent_workdir,
                            thread_id=codex_thread_id,
                        )
                        codex_thread_id = call_result.get("thread_id")
                    elif use_claude_print:
                        command_for_turn = _claude_print_command(
                            prompt, claude_session_id, resolved_model,
                            resume_session=iteration > 1,
                        )
                        call_result = _run_claude_print_turn(
                            prompt=prompt,
                            timeout_sec=timeout_sec,
                            session_id=claude_session_id,
                            model=resolved_model,
                            workdir=agent_workdir,
                            resume_session=iteration > 1,
                        )
                    else:
                        call_result = _session_send_and_receive(
                            proc=session_proc,
                            prompt=prompt,
                            timeout_sec=timeout_sec,
                        )
                        call_result["command"] = launch_cmd

                    response = call_result["response"]
                    turn_elapsed = call_result.get("elapsed_seconds")
                    turn_attempts = call_result.get("attempts", 1)
                    provider_fatal = _provider_fatal_kind("\n".join((
                        response,
                        str(call_result.get("stderr") or ""),
                    )))
                    if provider_fatal:
                        raise AgentTurnError(
                            response, command_for_turn or [agent],
                            failure_kind=provider_fatal, elapsed_seconds=turn_elapsed,
                        )
                    dialogue_turns.append(call_result)
                    query_rules = extract_query(response)
                    full_dl = compose_full_dl(task["schema_def"], query_rules, case_id=cid)
                except Exception as exc:
                    response = str(exc)
                    inference_ok = False
                    failure_kind = getattr(exc, "failure_kind", "agent_error")
                    provider_fatal = _provider_fatal_kind(response)
                    if provider_fatal:
                        failure_kind = provider_fatal
                    turn_elapsed = getattr(exc, "elapsed_seconds", None)
                    if hasattr(exc, "command") and isinstance(getattr(exc, "command"), list):
                        command_for_turn = getattr(exc, "command")
                    dialogue_turns.append(
                        {
                            "command": command_for_turn if command_for_turn else [agent],
                            "prompt": prompt,
                            "response": response,
                            "stderr": response,
                            "inference_ok": False,
                            "failure_kind": failure_kind,
                            "elapsed_seconds": turn_elapsed,
                            "attempts": turn_attempts,
                        }
                    )
                    query_rules = ""
                    full_dl = compose_full_dl(task["schema_def"], "", case_id=cid)

                    err_text = response.lower()
                    if isinstance(exc, AgentTurnError) or (
                        "returned empty output" in err_text
                        or "exited early" in err_text
                        or "persistent session stdin unavailable" in err_text
                        or "timed out" in err_text
                    ):
                        session_unhealthy = True
                    if failure_kind in {"quota", "account_disabled"}:
                        fatal_stop_kind = failure_kind

                if inference_ok:
                    eval_result, metric, perfect_match = _evaluate_query(
                        case_record=case_record,
                        full_dl=full_dl,
                        evaluate_query_file_for_case=evaluate_query_file_for_case,
                        case_demo_dirs=case_demo_dirs,
                        calculate_from_confusion=calculate_from_confusion,
                    )
                    round_feedback = "" if perfect_match else _feedback_from_result(eval_result)
                else:
                    # A provider/CLI failure is not a model-generated empty
                    # program. In particular, do not execute schema-only text:
                    # Souffle may compile it and turn a 402 into compile_ok=true.
                    eval_result = {
                        "compile_ok": False,
                        "tp": 0,
                        "fp": 0,
                        "fn": 0,
                    }
                    metric = {"precision": 0.0, "recall": 0.0, "f1": 0.0}
                    perfect_match = False
                    round_feedback = f"terminal inference failure: {failure_kind}"

                round_info = {
                    "iteration": iteration,
                    "selection_pool": "demo",
                    "compile_ok": eval_result["compile_ok"],
                    "perfect_match": perfect_match,
                    "precision": metric["precision"],
                    "recall": metric["recall"],
                    "f1": metric["f1"],
                    "tp": eval_result["tp"],
                    "fp": eval_result["fp"],
                    "fn": eval_result["fn"],
                    "feedback": round_feedback,
                    "query": query_rules,
                    "response": response,
                    "dialogue_turn": dialogue_turns[-1],
                    "inference_ok": inference_ok,
                    "failure_kind": failure_kind,
                    "elapsed_seconds": turn_elapsed,
                    "attempts": turn_attempts,
                }
                trace.append(round_info)

                if inference_ok and (
                    best is None or _best_round([best, round_info]) is round_info
                ):
                    best = {
                        "query": query_rules,
                        "full_dl": full_dl,
                        "iteration": iteration,
                        "compile_ok": eval_result["compile_ok"],
                        "perfect_match": perfect_match,
                        "precision": metric["precision"],
                        "recall": metric["recall"],
                        "f1": metric["f1"],
                        "tp": eval_result["tp"],
                        "fp": eval_result["fp"],
                        "fn": eval_result["fn"],
                        "feedback": round_feedback,
                        "response": response,
                    }

                print(
                    f"[AGENT:{agent}][run {run_id}] {cid} iter={iteration}/{max_iterations} "
                    f"compile_ok={eval_result['compile_ok']} f1={metric['f1']:.4f}"
                )

                if perfect_match:
                    break
                if session_unhealthy:
                    print(f"[WARN] {cid} session unhealthy ({failure_kind}), "
                          "stop remaining turns early.")
                    break
                feedback = round_feedback
        finally:
            if session_proc is not None:
                _stop_agent_session(session_proc)
            agent_workdir_ctx.cleanup()

        if best is None:
            best = {
                "query": "",
                "full_dl": compose_full_dl(task["schema_def"], "", case_id=cid),
                "iteration": 0,
                "compile_ok": False,
                "perfect_match": False,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "tp": 0,
                "fp": 0,
                "fn": 0,
                "feedback": "no candidate generated",
                "response": "",
            }

        query_path = run_dir / f"{cid}.dl"
        query_path.write_text(best["full_dl"], encoding="utf-8")

        trace_path = run_dir / f"{cid}.trace.json"
        with trace_path.open("w", encoding="utf-8") as f:
            json.dump(trace, f, ensure_ascii=False, indent=2)

        dialogue_path = run_dir / f"{cid}.dialogue.json"
        with dialogue_path.open("w", encoding="utf-8") as f:
            json.dump(dialogue_turns, f, ensure_ascii=False, indent=2)

        row = _agent_row(
            cid=cid,
            run_id=run_id,
            best=best,
            trace=trace,
            resolved_model=resolved_model,
            agent=agent,
            method=method,
            temperature=temperature,
            prompt_digest=prompt_digest,
            max_iterations=max_iterations,
            timeout_sec=timeout_sec,
            agent_cmd=agent_cmd,
            checkpoint_origin="protocol-v5-new",
        )
        _atomic_json(run_dir / f"{cid}.agent.json", row)
        rows.append(row)

        if not row["checkpoint_valid"]:
            failure = {
                "id": cid,
                "kind": row["failure_kind"] or "invalid_checkpoint",
                "error": row["inference_error"],
                "attempts": row["attempts"],
            }
            failures.append(failure)
            print(f"[AGENT-FAIL][run {run_id}] {cid}: {failure['kind']} "
                  f"after {failure['attempts']} attempt(s)")
        if fatal_stop_kind:
            print(f"[PROVIDER-FATAL][run {run_id}] {fatal_stop_kind}: stopping "
                  f"immediately after {cid}; remaining cases were not attempted")
            break

    aggregate_file = f"{case_id}.json" if case_id else f"{dataset_scope}.json"
    aggregate_path = run_dir / aggregate_file
    attempted_ids = {row["id"] for row in rows}
    missing = [task["case_id"] for task in tasks if task["case_id"] not in attempted_ids]
    complete = not failures and not missing
    if complete:
        _atomic_json(aggregate_path, rows)
        (run_dir / "progress.json").unlink(missing_ok=True)
        print(f"[DONE][run {run_id}] aggregate -> {aggregate_path}")
    else:
        _atomic_json(run_dir / "progress.json", rows)
        print(f"[INCOMPLETE][run {run_id}] failures={len(failures)} "
              f"not_attempted={len(missing)}; rerun with --resume")

    # Synthesis can tell a transport failure from a provider failure, but it
    # cannot tell whether the retry budget is spent -- that is a fact about the
    # sequence of runs, not about this one.  So it sorts by kind only, and
    # leaves the promotion of `retryable_failures` into scored terminal zeros
    # to agent_finalize_measurement.py.  The evaluator refuses a manifest that
    # has not been through the finalizer, so nothing here can be scored by
    # accident.
    retryable_failures = [f for f in failures
                          if measurement_v1.is_retryable_kind(f.get("kind"))]
    invalid_failures = [f for f in failures
                        if not measurement_v1.is_retryable_kind(f.get("kind"))]
    _atomic_json(run_dir / "agent_failures.json", {
        "agent": agent,
        "model": resolved_model,
        "endpoint": agent_container.agent_endpoints().get(agent),
        "credential_env": agent_container.AGENT_API_KEY_ENV,
        "resume_source": os.environ.get("AGENT_RESUME_SOURCE", "protocol-v5-run"),
        "method": method,
        "run": run_id,
        "agent_protocol_version": AGENT_PROTOCOL_VERSION,
        "complete_case_set": not missing,
        "complete": complete,
        "quota_exhausted": fatal_stop_kind == "quota",
        "fatal_error_kind": fatal_stop_kind,
        "failures": failures,
        "retryable_failures": retryable_failures,
        "invalid_failures": invalid_failures,
        "not_attempted": missing,
    })
    return {
        "complete": complete,
        "quota_exhausted": fatal_stop_kind == "quota",
        "fatal_error_kind": fatal_stop_kind,
    }


def main():
    parser = argparse.ArgumentParser(description="Synthesize Datalog queries via terminal coding agents with iterative validation.")
    parser.add_argument("--dataset", type=str, default="all", help="Case filter value from category, sub category, or all")
    parser.add_argument("--agent", type=str, required=True, help="Coding agent command label: codex or claude")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=(
            "Base model for coding agent CLI. "
            + _model_inventory().agent_default_help()
        ),
    )
    parser.add_argument(
        "--agent_cmd",
        type=str,
        default=None,
        help=(
            "Optional launch command template for persistent agent session. "
            "Supports placeholder: {agent}. "
            "If omitted, default command is agent itself. "
            "Example: \"{agent} --dangerously-bypass-approvals-and-sandbox\""
        ),
    )
    parser.add_argument("--method", type=str, default="signature", choices=["signature", "description"], help="Prompt strategy")
    parser.add_argument("--case_id", type=str, default=None, help="Optional single case id")
    parser.add_argument("--max_iterations", type=int, default=4, help="Max dialogue turns, including the initial attempt (formal matrix: 4)")
    parser.add_argument("--timeout_sec", type=int, default=60, help="Timeout for one agent invocation")
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help=(
            "Recorded decoding temperature for the pass@1 protocol (default 0). "
            "Note: the codex/claude CLIs do not expose a temperature flag, so this is "
            "provenance metadata; enforce it via the agent's endpoint/config (see test.py-style setup)."
        ),
    )
    parser.add_argument("--num_runs", type=int, default=1, help="Independent repetitions; each run writes to run_<i>/")
    parser.add_argument(
        "--resume", action="store_true",
        help=("Reuse exact valid protocol-v5 case checkpoints. Older checkpoints "
              "used the scored eval pool for adaptive feedback and are never reused."),
    )
    parser.add_argument(
        "--no-container", dest="container", action="store_false", default=True,
        help=(
            "Run the agent CLI on the host instead of inside the isolation "
            "container. The host setting was measured to leak: `workspace-write` "
            "restricts writes and network, not reads, so a reference program is "
            "readable by absolute path. Use only for debugging, and do not report "
            "numbers produced this way as agent results."
        ),
    )
    args = parser.parse_args()

    global _CONTAINER_MODE
    _CONTAINER_MODE = args.container
    if _CONTAINER_MODE:
        _require_container()
        prov = agent_container.image_provenance()
        print(f"[container] {prov['image']} ({prov['image_id']})")
        print(f"[container] codex={prov['codex_version']} claude={prov['claude_version']} "
              f"souffle={prov['souffle_version']}")
    else:
        print("[container] DISABLED -- agent runs on the host, reference programs are readable")

    return synthesize_with_coding_agent(
        dataset=args.dataset,
        agent=args.agent,
        model=args.model,
        method=args.method,
        case_id=args.case_id,
        max_iterations=args.max_iterations,
        timeout_sec=args.timeout_sec,
        agent_cmd=args.agent_cmd,
        temperature=args.temperature,
        num_runs=args.num_runs,
        resume=args.resume,
    )


if __name__ == "__main__":
    raise SystemExit(main())
