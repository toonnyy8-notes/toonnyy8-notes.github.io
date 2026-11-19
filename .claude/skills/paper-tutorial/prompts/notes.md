You are doing step 6-2 ("Research 整理資料") for a beginner-oriented interactive tutorial on one paper. Write in Traditional Chinese (台灣用語), plain literal academic style: no rhetorical questions, no metaphors, no promotional adjectives.

Paper: "{TITLE}" (arXiv {ARXIV}). Full text: {DIR}/src/paper.txt (reference list at the end).

Read first:
- {REPO}/AGENTS.md — section 「研究筆記（高內聚、低耦合）」 and the frontmatter spec.
- Existing notes to reuse instead of duplicating: {REPO}/scripts/notes_split/ (index.json lists them; grep titles and arxiv fields). If a concept already has a note, link it as `[標題](slug.html)`; do not rewrite it.

Task: trace the research lineage a reader needs to understand this paper step by step, using the paper's own references as the backbone. Verify every cited work's title, authors, year and arXiv ID via the arXiv API (curl "https://export.arxiv.org/api/query?id_list=ID" or search_query=ti:...); leave out anything you cannot verify. Claims about prior work come from that work's abstract or from how this paper describes it — record which in `sources`.

Output (create the folder) {DIR}/notes/:
1. One hub note (kind: hub) summarizing the lineage in reading order and linking each topic note.
2. Topic notes (kind: topic), one concept per note, roughly 4–8 notes, each 1–5 KB, frontmatter exactly per AGENTS.md (slug, title, cluster, kind, tags, arxiv, sources, related). `related` lists every slug the body links to. Every note's `arxiv` includes {ARXIV} if the note discusses this paper. Slugs must not collide with files in scripts/notes_split/.
3. `timeline.json`: list of {"year","title","authors_short","arxiv","role","source"} in chronological order, 8–15 verified entries, ending with this paper.

Report back in ≤8 lines: files created, existing notes linked, anything unverified.
