# LangSmith Evaluation Checkpoint

Branch: `eval/langsmith-team-project`

Current status:
- Added a LangSmith evaluation scaffold under `eval/`.
- Added dataset generation and audit prompts.
- Added a sample JSONL dataset example.
- Added dataset JSONL validation helpers.
- Added LangSmith dataset upload script.
- Added LangSmith evaluation runner.
- Added rule-based evaluators for JSON parse/schema, question count, and response length.
- Added an LLM-as-a-Judge evaluator.
- Added a target wrapper around `app.api.endpoints.chat.process_chat`.
- Added `langsmith` to `requirements.txt`.
- Updated JSON hard-gate evaluators to validate the actual chat API response
  (`content`, `suggestedQuestions`, `currentStatus`, `isSufficient`,
  `collectedData`) instead of incorrectly requiring `outputs.content` to be
  raw `project_progress_v1` JSON.

Validated:
- `python -c "from eval.langsmith.io import load_jsonl; print(len(load_jsonl('eval/datasets/generated/team_project_v1.sample.jsonl')))"` returned `1`.
- `python -m compileall eval` completed successfully.
- `python -c "from eval.langsmith.target import invoke_chat; print(callable(invoke_chat))"` returned `True` with a 60 second timeout.
- Local evaluator check returned `json_parse_pass=1` and `json_schema_pass=1`
  for a valid chat API response object.

Not run yet:
- `pip install -r requirements.txt`
- Dataset generation with a real OpenAI model.
- LangSmith dataset upload.
- LangSmith evaluation run.

Next commands:

```powershell
pip install -r requirements.txt
```

```powershell
python -m eval.langsmith.generate_dataset `
  --count 50 `
  --model gpt-4o `
  --output eval/datasets/generated/team_project_v1.jsonl
```

```powershell
python -c "from eval.langsmith.io import load_jsonl; print(len(load_jsonl('eval/datasets/generated/team_project_v1.jsonl')))"
```

```powershell
python -m eval.langsmith.create_dataset `
  --source eval/datasets/generated/team_project_v1.jsonl `
  --name matchatonic-team-project-v1
```

```powershell
python -m eval.langsmith.run_eval `
  --dataset matchatonic-team-project-v1 `
  --branch eval/langsmith-team-project
```

Required environment variables:
- `OPENAI_API_KEY`
- `LANGSMITH_API_KEY`
- `LANGSMITH_TRACING=true`
- Optional: `LANGSMITH_PROJECT`
- Optional: `LANGSMITH_JUDGE_MODEL`

Important note:
- Keep the generated dataset fixed when comparing branches. Run the same LangSmith dataset against each branch and compare experiments by branch/commit metadata.
