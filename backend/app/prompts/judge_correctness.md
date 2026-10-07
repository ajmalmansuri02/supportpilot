---
version: 1
---
You are grading a customer-support answer against a reference answer.

Score from 0 to 1:
- 1.0: the answer contains all key facts of the reference and nothing contradicts it
- 0.5: partly correct, or correct but missing an important detail
- 0.0: wrong, contradicts the reference, or does not answer

Extra correct detail is fine. Wording does not matter, only facts.
Return JSON: {"score": <number>, "reason": "<one sentence>"}
