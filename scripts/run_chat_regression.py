import argparse
import json
import sys
import urllib.error
import urllib.request


CASES = [
    {
        "name": "01_request_like_not_title",
        "notes": [
            "collectedData.title 가 생기면 실패",
            "응답은 미정 항목 안내 또는 다음 질문 유도여야 함",
        ],
        "payload": {
            "roomId": 2001,
            "content": "지금 미정인 항목이랑 제대로 정의되지 않은 부분까지 다 채워줘",
            "actionType": "CHAT",
            "currentStatus": "EXPLORE",
            "collectedData": {},
            "recentMessages": [],
            "selectedMessage": None,
            "selectedAnswers": [],
        },
    },
    {
        "name": "02_title_capture",
        "notes": [
            "collectedData.title 이 공공화장실 실시간 혼잡 안내 여야 함",
            "status 는 최소 TOPIC_SET 이상이어야 함",
        ],
        "payload": {
            "roomId": 2002,
            "content": "제목은 공공화장실 실시간 혼잡 안내로 할래",
            "actionType": "CHAT",
            "currentStatus": "EXPLORE",
            "collectedData": {},
            "recentMessages": [],
            "selectedMessage": None,
            "selectedAnswers": [],
        },
    },
    {
        "name": "03_goal_request_after_title",
        "notes": [
            "초기 탐색 반복이 아니라 title 기반 goal 정리로 가야 함",
        ],
        "payload": {
            "roomId": 2003,
            "content": "이 제목에 맞는 프로젝트 목표를 세워줘",
            "actionType": "CHAT",
            "currentStatus": "TOPIC_SET",
            "collectedData": {
                "title": "공공화장실 실시간 혼잡 안내",
            },
            "recentMessages": [
                "사용자: 공공시설 관련 서비스로 하고 싶어",
                "AI: 어떤 시설을 대상으로 하면 좋을까요?",
            ],
            "selectedMessage": None,
            "selectedAnswers": [],
        },
    },
    {
        "name": "04_team_size",
        "notes": [
            "collectedData.teamSize 가 4 여야 함",
        ],
        "payload": {
            "roomId": 2004,
            "content": "4명이야",
            "actionType": "CHAT",
            "currentStatus": "GATHER",
            "collectedData": {
                "title": "공공화장실 실시간 혼잡 안내",
                "goal": "사용자가 혼잡한 공공화장실을 피해서 더 편하게 이용할 수 있도록 돕는다",
            },
            "recentMessages": [],
            "selectedMessage": None,
            "selectedAnswers": [],
        },
    },
    {
        "name": "05_roles",
        "notes": [
            "collectedData.roles 가 개발자, 기획자, PM 계열로 반영되어야 함",
        ],
        "payload": {
            "roomId": 2005,
            "content": "개발자, 기획자, PM으로 나눌 거야",
            "actionType": "CHAT",
            "currentStatus": "GATHER",
            "collectedData": {
                "title": "공공화장실 실시간 혼잡 안내",
                "goal": "사용자가 혼잡한 공공화장실을 피해서 더 편하게 이용할 수 있도록 돕는다",
                "teamSize": 4,
            },
            "recentMessages": [],
            "selectedMessage": None,
            "selectedAnswers": [],
        },
    },
    {
        "name": "06_summary_no_mutation",
        "notes": [
            "returned collectedData 가 입력과 동일해야 함",
            "summary 내용이 state 와 일치해야 함",
        ],
        "payload": {
            "roomId": 2006,
            "content": "지금까지 모인 정보 요약해줘",
            "actionType": "CHAT",
            "currentStatus": "GATHER",
            "collectedData": {
                "title": "공공화장실 실시간 혼잡 안내",
                "goal": "사용자가 혼잡한 공공화장실을 피해서 더 편하게 이용할 수 있도록 돕는다",
                "teamSize": 4,
                "roles": ["개발자", "기획자", "PM"],
            },
            "recentMessages": [],
            "selectedMessage": None,
            "selectedAnswers": [],
        },
    },
    {
        "name": "07_request_next_step",
        "notes": [
            "남은 필드(deadline, deliverables 등)를 물어봐야 함",
        ],
        "payload": {
            "roomId": 2007,
            "content": "그럼 다음으로 뭐 정하면 돼?",
            "actionType": "CHAT",
            "currentStatus": "GATHER",
            "collectedData": {
                "title": "공공화장실 실시간 혼잡 안내",
                "goal": "사용자가 혼잡한 공공화장실을 피해서 더 편하게 이용할 수 있도록 돕는다",
                "teamSize": 4,
                "roles": ["개발자", "기획자", "PM"],
            },
            "recentMessages": [],
            "selectedMessage": None,
            "selectedAnswers": [],
        },
    },
    {
        "name": "08_mixed_utterance",
        "notes": [
            "teamSize, roles 는 반영되고 응답은 다음 미정 항목으로 넘어가야 함",
        ],
        "payload": {
            "roomId": 2008,
            "content": "팀원은 4명이고 역할은 개발자, 기획자, PM 정도로 할 것 같아. 다음엔 뭐 정해야 해?",
            "actionType": "CHAT",
            "currentStatus": "GATHER",
            "collectedData": {
                "title": "공공화장실 실시간 혼잡 안내",
                "goal": "사용자가 혼잡한 공공화장실을 피해서 더 편하게 이용할 수 있도록 돕는다",
            },
            "recentMessages": [],
            "selectedMessage": None,
            "selectedAnswers": [],
        },
    },
    {
        "name": "09_overwrite_team_size",
        "notes": [
            "teamSize 가 5 로 바뀌어야 함",
        ],
        "payload": {
            "roomId": 2009,
            "content": "아니 팀원은 4명이 아니라 5명이야",
            "actionType": "CHAT",
            "currentStatus": "GATHER",
            "collectedData": {
                "title": "공공화장실 실시간 혼잡 안내",
                "goal": "사용자가 혼잡한 공공화장실을 피해서 더 편하게 이용할 수 있도록 돕는다",
                "teamSize": 4,
                "roles": ["개발자", "기획자", "PM"],
            },
            "recentMessages": [],
            "selectedMessage": None,
            "selectedAnswers": [],
        },
    },
    {
        "name": "10_request_like_not_goal",
        "notes": [
            "goal 이나 title 에 요청형 문장이 저장되면 실패",
        ],
        "payload": {
            "roomId": 2010,
            "content": "일단 제대로 안 정해진 것들부터 정리해줘",
            "actionType": "CHAT",
            "currentStatus": "GATHER",
            "collectedData": {
                "title": "공공화장실 실시간 혼잡 안내",
            },
            "recentMessages": [],
            "selectedMessage": None,
            "selectedAnswers": [],
        },
    },
]


def post_json(url: str, payload: dict) -> tuple[int, dict]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        status = response.getcode()
        text = response.read().decode("utf-8")
        return status, json.loads(text)


def print_case_result(index: int, case: dict, status: int, response: dict, *, compact: bool) -> None:
    print(f"\n[{index:02d}] {case['name']}")
    print("notes:")
    for note in case["notes"]:
        print(f"- {note}")
    print(f"status_code: {status}")
    print(f"request.content: {case['payload']['content']}")
    if compact:
        print("response.summary:")
        print(json.dumps(
            {
                "content": response.get("content"),
                "currentStatus": response.get("currentStatus"),
                "isSufficient": response.get("isSufficient"),
                "collectedData": response.get("collectedData"),
            },
            ensure_ascii=False,
            indent=2,
        ))
        return

    print("request.payload:")
    print(json.dumps(case["payload"], ensure_ascii=False, indent=2))
    print("response.body:")
    print(json.dumps(response, ensure_ascii=False, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser(description="Run 10 manual regression requests against /ai/chat/.")
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000",
        help="Base server URL, for example http://127.0.0.1:8000 or http://15.164.221.247:8000",
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Print only response summary fields.",
    )
    args = parser.parse_args()

    url = args.base_url.rstrip("/") + "/ai/chat/"
    print(f"target: {url}")

    failures = 0
    for index, case in enumerate(CASES, start=1):
        try:
            status, response = post_json(url, case["payload"])
            print_case_result(index, case, status, response, compact=args.compact)
        except urllib.error.HTTPError as exc:
            failures += 1
            body = exc.read().decode("utf-8", errors="replace")
            print(f"\n[{index:02d}] {case['name']}")
            print(f"HTTPError: {exc.code}")
            print(body)
        except Exception as exc:  # pragma: no cover - manual script
            failures += 1
            print(f"\n[{index:02d}] {case['name']}")
            print(f"Error: {exc}")

    if failures:
        print(f"\ncompleted with {failures} request failure(s)")
        return 1

    print("\ncompleted without transport errors")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
