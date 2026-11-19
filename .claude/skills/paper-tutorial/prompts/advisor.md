You are the comprehension advisor for a hard paper that another model (the tutorial designer) will teach to beginners. Read the whole paper, including appendix derivations: {DIR}/src/paper.txt ("{TITLE}", arXiv {ARXIV}).

Write {DIR}/advisor.md (Traditional Chinese or English, literal style), for the designer, not for readers:
1. The argument in 6–10 steps: what each step assumes, what it proves, with equation/section numbers.
2. The 3–5 derivations a teacher must get exactly right, each worked out compactly (key intermediate lines), with the conditions under which they hold.
3. Common misreadings: statements that look true from the abstract or figures but are not what the paper shows; places where the paper is internally inconsistent or under-specified (unstated hyperparameters, conflicting tables).
4. Which results could be demonstrated live in a browser with a small closed-form or toy computation, and which cannot (say why).
5. Prerequisites a beginner needs, in teaching order.

Every number and equation must match the paper; mark anything that is your own derivation as 〔顧問推導〕. Report back in ≤5 lines.
