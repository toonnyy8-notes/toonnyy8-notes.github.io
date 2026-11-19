Readiness check: GO. Now build the page. You hold the intent, so you write all section prose yourself (Traditional Chinese, literal academic style per AGENTS.md).

First apply the readiness report's non-blocking notes that are cheap, recording them in design.md.

Build {DIR}/build/index.html as one self-contained file. Per-section check questions go in `<details class="check">` (the acceptance script strips exactly these); other collapsibles must not use that class. Start from the shell of {REPO}/tutorials/prediction-targets-k-diff.html (copy its CSS tokens, #pm-bar nav, section TOC, plate/callout styles, KaTeX setup and `?test` selfTest harness) and replace only the content and widgets; do not redesign the layout. Embed {DIR}/notes/timeline.json; note links are `<slug>.html`; paper link `../index.html#paper-{ID}`.

Verify before reporting:
1. LO self-check: every learning objective's reference answer appears in body prose. Run `python3 {REPO}/.claude/skills/paper-tutorial/scripts/extract_page_text.py {DIR}` and confirm each answer's key terms and numbers are present in {DIR}/accept/page.txt (answer boxes are stripped there). Fix any gap in prose.
2. `?test` passes all acceptance tables: google-chrome --headless=new --disable-gpu --no-sandbox --virtual-time-budget=9000 --dump-dom "file://{DIR}/build/index.html?test" and read the result written into the title.
3. One visual round only: screenshot the new widgets at 1440 and the whole page at 390 (google-chrome --headless=new --disable-gpu --no-sandbox --hide-scrollbars --virtual-time-budget=6000 --window-size=W,H --screenshot=… file://…), plus one --force-dark-mode capture; fix what is broken. Save to {DIR}/build/caps/.
4. `npx -y impeccable detect {DIR}/build/index.html` → 0 anti-patterns (numbered-section-label advisories are allowed).
5. No horizontal scroll at 390px; KaTeX renders; no console errors.

Report back in ≤6 lines: selfTest result, LO self-check, impeccable result, open issues.
