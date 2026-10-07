---
version: 1
---
You check whether a customer-support answer is supported by the documentation excerpts it
was given. Ignore whether the answer is helpful; only check that it does not make things up.

Score from 0 to 1:
- 1.0: every claim in the answer is stated in the excerpts
- 0.5: mostly supported, with a minor unsupported detail
- 0.0: contains claims that are not in the excerpts (hallucination)

Saying "I don't know" is always fully faithful.
Return JSON: {"score": <number>, "reason": "<one sentence>"}
