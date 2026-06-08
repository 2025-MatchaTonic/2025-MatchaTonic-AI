from __future__ import annotations

import argparse

from dotenv import load_dotenv
from langsmith import Client
from langsmith.utils import LangSmithNotFoundError

from eval.langsmith.io import load_jsonl


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Upload a JSONL evaluation dataset to LangSmith.")
    parser.add_argument("--source", required=True, help="Path to JSONL dataset.")
    parser.add_argument("--name", required=True, help="LangSmith dataset name.")
    parser.add_argument("--description", default="MatchaTonic team project chat evaluation dataset.")
    args = parser.parse_args()

    examples = load_jsonl(args.source)
    client = Client()

    try:
        dataset = client.read_dataset(dataset_name=args.name)
        print(f"Using existing dataset: {dataset.name}")
    except LangSmithNotFoundError:
        dataset = client.create_dataset(dataset_name=args.name, description=args.description)
        print(f"Created dataset: {dataset.name}")

    client.create_examples(
        dataset_id=dataset.id,
        inputs=[example["inputs"] for example in examples],
        outputs=[example["outputs"] for example in examples],
        metadata=[example["metadata"] for example in examples],
    )
    print(f"Uploaded {len(examples)} examples to {args.name}")


if __name__ == "__main__":
    main()
