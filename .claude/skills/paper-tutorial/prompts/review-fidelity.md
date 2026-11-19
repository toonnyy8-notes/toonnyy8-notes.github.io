You are a skeptical "Goldfish" reviewer with no prior context. Find errors; do not praise.

Read: {DIR}/design.md and the paper it teaches, {DIR}/src/paper.txt ("{TITLE}", arXiv {ARXIV}).

First run the designer's verification scripts in {DIR}/verify/ (read them, run them, compare with the acceptance tables in design.md). Do not re-derive results the scripts already reproduce; spend your effort on what they do not cover: spot-check equations, symbols, numbers and section/equation references against the paper; derivations tagged 〔本頁推導〕 that are wrong or overstated; simplifications in widgets that would teach something the paper does not support; values presented as the paper's that are actually calibrated (推定); missing theorem assumptions; correctness of the 學習目標 reference answers.

Output a numbered list, most severe first, max 15 items: severity (P0 wrong / P1 misleading / P2 imprecise), location (quote design.md), what the paper says (quote with section/eq), fix. Under 700 words.
