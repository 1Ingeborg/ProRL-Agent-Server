"""Thin launcher: upstream SWE-agent, SWE-ReX local, preinstalled editor dependencies."""
import argparse, copy, json, os, shutil, subprocess, sys, time, traceback
from pathlib import Path
import yaml

ROOT = Path(os.environ.get("POLAR_HOME", "/data/ProRL-Agent-Serve"))
SOURCE = ROOT / "repos/SWE-agent"

def qwen_prompt_ids(tokenizer, messages, tools=None):
    """Render the same normalized text/tool request as the pinned SGLang API."""
    messages = copy.deepcopy(messages)
    systems, turns = [], []
    for message in messages:
        if message["role"] in {"system", "developer"}:
            content = message.get("content") or ""
            if isinstance(content, list):
                content = "".join(part.get("text", "") for part in content)
            if content:
                systems.append(content)
        else:
            message["content"] = message.get("content") or ""
            for call in message.get("tool_calls") or []:
                arguments = call["function"].get("arguments")
                if isinstance(arguments, str):
                    call["function"]["arguments"] = json.loads(arguments)
            turns.append(message)
    if systems:
        turns.insert(0, {"role": "system", "content": "\n\n".join(systems)})
    normalized_tools = None
    if tools:
        normalized_tools = []
        for tool in tools:
            function = tool["function"]
            # Keep SGLang Function/Tool field order and defaults: these fields
            # are serialized into the Qwen tool prompt and consume tokens.
            normalized_function = {
                "description": function.get("description"),
                "name": function["name"],
                "parameters": function.get("parameters"),
                "strict": function.get("strict", False),
            }
            deferred = function.get("defer_loading", tool.get("defer_loading"))
            if deferred is not None:
                normalized_function["defer_loading"] = deferred
            normalized_tools.append({"type": tool.get("type", "function"),
                "function": normalized_function, "defer_loading": tool.get("defer_loading")})
    encoded = tokenizer.apply_chat_template(turns, tools=normalized_tools,
        tokenize=True, add_generation_prompt=True, enable_thinking=False)
    return encoded["input_ids"] if hasattr(encoded, "keys") else encoded

def stage_tools(target):
    target = Path(target)
    target.mkdir(parents=True, exist_ok=True)
    bundle = target / "edit_anthropic"
    shutil.copytree(SOURCE / "tools/edit_anthropic", bundle, dirs_exist_ok=True)
    editor = bundle / "bin/str_replace_editor"
    lines = editor.read_text().splitlines(True)
    lines[0] = "#!" + str(ROOT / "conda/swe/bin/python") + "\n"
    editor.write_text("".join(lines))
    editor.chmod(0o755)
    # Only installation changes: the editor body stays byte-identical upstream.
    (bundle / "install.sh").write_text(
        str(ROOT / "conda/swe/bin/python") +
        " -c 'from tree_sitter_languages import get_parser; get_parser(\"python\")'\n")
    return bundle

def build_config(instruction, model, api_base, api_key, output_dir, call_limit=80, tool_bundle=None, max_input_tokens=122880, max_output_tokens=8192, temperature=0.7):
    from sweagent.run.run_single import RunSingleConfig
    config = yaml.safe_load((SOURCE / "config/default.yaml").read_text())
    # Tool output was ~63% of trajectory tokens; clip each observation to 20k chars.
    config["agent"]["templates"]["max_observation_length"] = 20000
    config["agent"]["model"] = dict(name="openai/" + model.rsplit("/", 1)[-1],
        api_base=api_base, api_key=api_key, per_instance_cost_limit=0,
        total_cost_limit=0, per_instance_call_limit=call_limit, max_input_tokens=max_input_tokens,
        max_output_tokens=max_output_tokens, temperature=temperature,
        completion_kwargs={"max_tokens": max_output_tokens, "stream": False, "timeout": 600,
                           "parallel_tool_calls": False,
                           "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}})
    if tool_bundle is not None:
        for bundle in config["agent"]["tools"]["bundles"]:
            if bundle["path"] == "tools/edit_anthropic":
                bundle["path"] = str(tool_bundle)
    config.update(env=dict(deployment=dict(type="local"),
        repo=dict(type="preexisting", repo_name="testbed", reset=False)),
        problem_statement=dict(type="text", text=instruction), output_dir=output_dir)
    return RunSingleConfig.model_validate(config)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--instruction", default="Offline configuration validation")
    p.add_argument("--run-id", default="unknown")
    p.add_argument("--model", default="Qwen/Qwen3.5-4B")
    p.add_argument("--call-limit", type=int, default=80)
    p.add_argument("--max-input-tokens", type=int, default=122880)
    p.add_argument("--max-output-tokens", type=int, default=8192)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--validate-only", action="store_true")
    p.add_argument("--gateway-socket")
    args = p.parse_args()
    os.chdir(SOURCE)
    if args.validate_only:
        build_config(args.instruction, args.model, "http://127.0.0.1:9/v1", "offline", "/tmp/swe-offline",
                     args.call_limit, max_input_tokens=args.max_input_tokens,
                     max_output_tokens=args.max_output_tokens, temperature=args.temperature)
        print("SWE-agent configuration valid; deployment=local; no model call")
        return
    if not Path("/.dockerenv").exists():
        raise RuntimeError("Real SWE-agent execution is allowed only inside the task container")
    endpoint = os.environ.get("OPENAI_BASE_URL", "")
    key = os.environ.get("OPENAI_API_KEY", "")
    if not endpoint or not key:
        raise RuntimeError("Polar model proxy URL and session key are required")
    bridge = None
    output = Path("/polar/session/logs/agent/swe-agent")
    output.mkdir(parents=True, exist_ok=True)
    status = {"status": "running", "real_agent": True, "run_id": args.run_id,
              "configured_call_limit": args.call_limit}
    request_attempts = 0
    try:
        if args.gateway_socket:
            if not Path(args.gateway_socket).is_socket():
                raise RuntimeError("Local Polar gateway bridge is not running")
            bridge = subprocess.Popen([sys.executable, "/polar/session/swe_gateway_bridge.py",
                "--mode", "tcp", "--socket", args.gateway_socket], stdout=subprocess.DEVNULL)
            for _ in range(50):
                import socket
                with socket.socket() as sock:
                    if sock.connect_ex(("127.0.0.1", 18080)) == 0:
                        break
                if bridge.poll() is not None:
                    raise RuntimeError("Container model bridge exited")
                time.sleep(0.1)
            else:
                raise RuntimeError("Container model bridge startup timed out")
            endpoint = "http://127.0.0.1:18080/v1"
        from sweagent.run.run_single import RunSingle
        bundle = stage_tools("/polar/session/swe-tool-bundles")
        config = build_config(args.instruction, args.model, endpoint, key, str(output),
                              args.call_limit, bundle, args.max_input_tokens, args.max_output_tokens, args.temperature)
        run = RunSingle.from_config(config)
        # Count the complete Qwen chat request (template, tools, and generation
        # prefix), matching the tokenizer and template used by the inference server.
        from transformers import AutoTokenizer
        import litellm
        tok = AutoTokenizer.from_pretrained("/polar/session/tokenizer", local_files_only=True)
        original_counter = litellm.utils.token_counter
        tool_config = run.agent.tools.config
        tool_specs = tool_config.tools if tool_config.use_function_calling else None
        def qwen_chat_tokens(messages, selected_tools):
            return len(qwen_prompt_ids(tok, messages, selected_tools))
        def exact_qwen_counter(model="", custom_tokenizer=None, text=None, messages=None,
                               count_response_tokens=False, tools=None, **kwargs):
            if messages is not None:
                return qwen_chat_tokens(messages, tools or tool_specs)
            if text is not None:
                if isinstance(text, list):
                    return sum(len(tok.encode(part, add_special_tokens=False)) for part in text)
                return len(tok.encode(text, add_special_tokens=False))
            return original_counter(model=model, custom_tokenizer=custom_tokenizer,
                text=text, messages=messages, count_response_tokens=count_response_tokens,
                tools=tools, **kwargs)
        litellm.utils.token_counter = exact_qwen_counter
        # Keep SWE-agent's imported litellm module on this exact adapter too.
        try:
            import sweagent.agent.models as swe_models
            swe_models.litellm.utils.token_counter = exact_qwen_counter
        except Exception:
            pass
        original_completion = litellm.completion
        token_audit = Path("/polar/session/logs/agent/token-count-audit.jsonl")
        def audited_completion(*call_args, **call_kwargs):
            nonlocal request_attempts
            messages = call_kwargs.get("messages")
            if messages is not None:
                expected = qwen_chat_tokens(messages, call_kwargs.get("tools") or tool_specs)
                if expected > args.max_input_tokens:
                    from sweagent.exceptions import ContextWindowExceededError
                    raise ContextWindowExceededError(
                        f"Input tokens {expected} exceed max tokens {args.max_input_tokens}")
            if request_attempts >= args.call_limit:
                from sweagent.exceptions import InstanceCallLimitExceededError
                raise InstanceCallLimitExceededError("Per instance call limit reached")
            request_attempts += 1
            response = original_completion(*call_args, **call_kwargs)
            if messages is not None:
                expected = qwen_chat_tokens(messages, call_kwargs.get("tools") or tool_specs)
                usage = getattr(response, "usage", None)
                actual = getattr(usage, "prompt_tokens", None) if usage is not None else None
                with token_audit.open("a") as audit:
                    audit.write(json.dumps({"run_id": args.run_id, "template_tokens": expected,
                        "server_prompt_tokens": actual, "match": actual == expected}) + "\n")
                if actual is not None and actual != expected:
                    raise RuntimeError(f"Qwen prompt token mismatch: local={expected}, server={actual}")
            return response
        litellm.completion = audited_completion
        result_holder = {}
        original_agent_run = run.agent.run
        def tracked_agent_run(*a, **kw):
            result = original_agent_run(*a, **kw)
            result_holder["result"] = result
            return result
        run.agent.run = tracked_agent_run
        try:
            run.run()
        finally:
            run.env.close()
        result = result_holder.get("result")
        subprocess.run(["git", "add", "-A"], cwd="/testbed", check=False, capture_output=True)
        patch = subprocess.run(["git", "diff", "--cached", "--binary", "HEAD"], cwd="/testbed",
                               capture_output=True, check=False).stdout
        model_stats = (result.info.get("model_stats", {}) if result else {})
        status.update(status="completed",
            exit_reason=(result.info.get("exit_status") if result else "missing_agent_result"),
            calls=(model_stats.get("api_calls") if isinstance(model_stats, dict) else getattr(model_stats, "api_calls", None)),
            tokens_sent=(model_stats.get("tokens_sent") if isinstance(model_stats, dict) else getattr(model_stats, "tokens_sent", None)),
            tokens_received=(model_stats.get("tokens_received") if isinstance(model_stats, dict) else getattr(model_stats, "tokens_received", None)),
            patch_bytes=len(patch), empty_patch=(not bool(patch.strip())))
        if status["exit_reason"] in {"exit_error", "missing_agent_result"}:
            raise RuntimeError("SWE-agent failed: " + str(status["exit_reason"]))
    except Exception:
        status.update(status="error", error=traceback.format_exc())
        raise
    finally:
        status["request_attempts"] = request_attempts
        (output / "launcher-status.json").write_text(json.dumps(status, indent=2))
        if bridge is not None:
            bridge.terminate()
            try: bridge.wait(timeout=5)
            except subprocess.TimeoutExpired:
                bridge.kill(); bridge.wait()

if __name__ == "__main__":
    main()
