import argparse
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


def _http_get(url: str, timeout: int = 3) -> int:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return response.getcode()


def _http_post(url: str, payload: dict, timeout: int = 20) -> tuple[int, dict]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.getcode(), json.loads(response.read().decode("utf-8"))


def _wait_until_ready(base_url: str, timeout_seconds: int = 30) -> None:
    deadline = time.time() + timeout_seconds
    last_error = None
    while time.time() < deadline:
        try:
            status = _http_get(base_url + "/")
            if status == 200:
                return
        except Exception as exc:
            last_error = exc
            time.sleep(0.5)
    raise RuntimeError(f"server did not become ready: {last_error}")


def _start_server(repo_root: Path, port: int, log_prefix: Path) -> subprocess.Popen:
    log_prefix.parent.mkdir(parents=True, exist_ok=True)
    stdout_handle = open(str(log_prefix) + ".log", "w", encoding="utf-8")
    stderr_handle = open(str(log_prefix) + ".err.log", "w", encoding="utf-8")
    command = [
        sys.executable,
        "scripts/run_replay_server.py",
        "--repo-root",
        str(repo_root),
        "--port",
        str(port),
    ]
    process = subprocess.Popen(
        command,
        cwd=Path(__file__).resolve().parents[1],
        stdout=stdout_handle,
        stderr=stderr_handle,
        text=True,
    )
    process._stdout_handle = stdout_handle
    process._stderr_handle = stderr_handle
    return process


def _stop_server(process: subprocess.Popen) -> None:
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)
    process._stdout_handle.close()
    process._stderr_handle.close()


def _write_json(path: Path, data: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _collect_response(base_url: str, payload: dict) -> dict:
    endpoint = base_url.rstrip("/") + "/ai/chat/"
    try:
        status_code, response = _http_post(endpoint, payload)
        return {
            "status_code": status_code,
            "ok": True,
            "response": response,
        }
    except urllib.error.HTTPError as exc:
        return {
            "status_code": exc.code,
            "ok": False,
            "error": {
                "type": "HTTPError",
                "message": exc.read().decode("utf-8", errors="replace"),
            },
        }
    except Exception as exc:
        return {
            "status_code": None,
            "ok": False,
            "error": {
                "type": exc.__class__.__name__,
                "message": str(exc),
            },
        }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run /ai/chat/ HTTP replay against current branch and main worktree without external LLM/RAG.",
    )
    parser.add_argument(
        "--case-dir",
        default="tests/manual_cases/chat",
        help="Directory containing replay payload JSON files.",
    )
    parser.add_argument(
        "--output-dir",
        default="tests/manual_cases/out/http_compare",
        help="Directory to write replay responses.",
    )
    parser.add_argument("--current-port", type=int, default=8051)
    parser.add_argument("--main-port", type=int, default=8050)
    parser.add_argument(
        "--main-root",
        default=".worktrees/main-compare",
        help="Path to the main branch worktree.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    main_root = (repo_root / args.main_root).resolve()
    case_dir = (repo_root / args.case_dir).resolve()
    output_dir = (repo_root / args.output_dir).resolve()

    current_commit = subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=repo_root,
        text=True,
    ).strip()
    main_commit = subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=main_root,
        text=True,
    ).strip()

    current_process = _start_server(
        repo_root,
        args.current_port,
        output_dir / "current_server",
    )
    main_process = _start_server(
        main_root,
        args.main_port,
        output_dir / "main_server",
    )

    try:
        _wait_until_ready(f"http://127.0.0.1:{args.current_port}")
        _wait_until_ready(f"http://127.0.0.1:{args.main_port}")

        current_dir = output_dir / f"current_{current_commit}"
        main_dir = output_dir / f"main_{main_commit}"
        summary: list[dict] = []

        for case_path in sorted(case_dir.glob("*.json")):
            payload = json.loads(case_path.read_text(encoding="utf-8"))
            case_name = case_path.stem
            current_result = _collect_response(
                f"http://127.0.0.1:{args.current_port}",
                payload,
            )
            main_result = _collect_response(
                f"http://127.0.0.1:{args.main_port}",
                payload,
            )
            _write_json(current_dir / f"{case_name}.json", current_result)
            _write_json(main_dir / f"{case_name}.json", main_result)
            summary.append(
                {
                    "case": case_name,
                    "current_status": current_result.get("status_code"),
                    "current_ok": current_result.get("ok"),
                    "main_status": main_result.get("status_code"),
                    "main_ok": main_result.get("ok"),
                    "current_error": current_result.get("error"),
                    "main_error": main_result.get("error"),
                }
            )

        _write_json(output_dir / "summary.json", summary)
        return 0
    finally:
        _stop_server(current_process)
        _stop_server(main_process)


if __name__ == "__main__":
    raise SystemExit(main())
