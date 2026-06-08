You are an AI evaluation dataset auditor.

Audit the provided JSONL dataset for LangSmith LLM-as-a-Judge evaluation.

Criteria:
1. Is the team chat realistic?
2. Are outputs.must_include items explicitly grounded in the chat?
3. Do outputs.must_not_include items help catch hallucination?
4. Is expected_behavior specific enough for a judge?
5. Are category, difficulty, split, and failure_mode appropriate?
6. Is each line valid JSON?
7. Are examples overly duplicated?
8. Is the dataset ready for LangSmith evaluation?

Output JSON only:
{
  "overall_quality_score": 1,
  "invalid_examples": [
    {
      "line": 1,
      "issue": "Problem description",
      "fix": "Concrete fix"
    }
  ],
  "duplicates": [
    {
      "lines": [1, 2],
      "reason": "Why they are duplicates"
    }
  ],
  "recommended_changes": ["Concrete changes"]
}

Dataset:
{JSONL_DATA}
