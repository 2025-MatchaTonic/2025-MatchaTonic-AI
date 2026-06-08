import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path


def post_json(url: str, payload: dict) -> tuple[int, dict]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        status = response.getcode()
        text = response.read().decode("utf-8")
        return status, json.loads(text)


def load_cases(case_dir: Path) -> list[Path]:
    return sorted(path for path in case_dir.glob("*.json") if path.is_file())


def summarize_response(response: dict) -> dict:
    return {
        "content": response.get("content"),
        "currentStatus": response.get("currentStatus"),
        "isSufficient": response.get("isSufficient"),
        "suggestedQuestions": response.get("suggestedQuestions"),
        "collectedData": response.get("collectedData"),
    }


def build_diff(left: dict, right: dict) -> dict:
    keys = sorted(set(left.keys()) | set(right.keys()))
    diff: dict[str, dict[str, object]] = {}
    for key in keys:
        if left.get(key) == right.get(key):
            continue
        diff[key] = {"baseline": left.get(key), "target": right.get(key)}
    return diff


def write_response(output_dir: Path, prefix: str, case_name: str, response: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{case_name}.{prefix}.json"
    path.write_text(json.dumps(response, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Replay /ai/chat/ payload files against a local or remote server.",
    )
    parser.add_argument(
        "--case-dir",
        default="tests/manual_cases/chat",
        help="Directory that contains request JSON files.",
    )
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000",
        help="Target server base URL.",
    )
    parser.add_argument(
        "--baseline-url",
        default="",
        help="Optional baseline server base URL for side-by-side comparison.",
    )
    parser.add_argument(
        "--endpoint",
        default="/ai/chat/",
        help="Endpoint path to replay.",
    )
    parser.add_argument(
        "--output-dir",
        default="tests/manual_cases/out",
        help="Directory for saved responses.",
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Print only summary fields instead of full JSON.",
    )
    args = parser.parse_args()

    case_dir = Path(args.case_dir)
    output_dir = Path(args.output_dir)
    endpoint = "/" + args.endpoint.strip("/")
    case_paths = load_cases(case_dir)

    if not case_paths:
        print(f"no case files found under {case_dir}", file=sys.stderr)
        return 1

    target_url = args.base_url.rstrip("/") + endpoint
    baseline_url = args.baseline_url.rstrip("/") + endpoint if args.baseline_url else ""

    print(f"target   : {target_url}")
    if baseline_url:
        print(f"baseline : {baseline_url}")
    print(f"cases    : {len(case_paths)}")

    failures = 0
    for index, case_path in enumerate(case_paths, start=1):
        payload = json.loads(case_path.read_text(encoding="utf-8"))
        case_name = case_path.stem
        print(f"\n[{index:02d}] {case_name}")

        try:
            target_status, target_response = post_json(target_url, payload)
            write_response(output_dir, "target", case_name, target_response)
            print(f"target_status: {target_status}")
            print(
                json.dumps(
                    summarize_response(target_response) if args.compact else target_response,
                    ensure_ascii=False,
                    indent=2,
                )
            )
        except urllib.error.HTTPError as exc:
            failures += 1
            body = exc.read().decode("utf-8", errors="replace")
            print(f"target HTTPError: {exc.code}")
            print(body)
            continue
        except Exception as exc:
            failures += 1
            print(f"target Error: {exc}")
            continue

        if not baseline_url:
            continue

        try:
            baseline_status, baseline_response = post_json(baseline_url, payload)
            write_response(output_dir, "baseline", case_name, baseline_response)
            print(f"baseline_status: {baseline_status}")
            diff = build_diff(
                summarize_response(baseline_response),
                summarize_response(target_response),
            )
            print("diff:")
            print(json.dumps(diff, ensure_ascii=False, indent=2))
        except urllib.error.HTTPError as exc:
            failures += 1
            body = exc.read().decode("utf-8", errors="replace")
            print(f"baseline HTTPError: {exc.code}")
            print(body)
        except Exception as exc:
            failures += 1
            print(f"baseline Error: {exc}")

    if failures:
        print(f"\ncompleted with {failures} transport failure(s)")
        return 1

    print("\ncompleted without transport errors")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
