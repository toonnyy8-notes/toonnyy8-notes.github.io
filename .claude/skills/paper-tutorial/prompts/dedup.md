# Research-note dedup

Notes dir: N={REPO}/scripts/notes_split
Rules: read the "研究筆記（高內聚、低耦合）" section of {REPO}/AGENTS.md.
Assigned files: {FILES}
Read-only hubs: {HUBS}
Mapping file: {MAP}

One topic is written once; other notes link to it with [標題](slug.html). Keep every fact, number, arXiv ID and citation exactly as written; add nothing new.

These notes were written in parallel for different papers' tutorials, so the same topic (a model, a method, an objective, a research lineage) is sometimes explained in two or more notes. Your job, for the files assigned to you:

1. Read every assigned file. Find **duplicate topics**: two notes whose main subject is the same method/model/lineage, or a section in one note that re-explains what another note's main subject already covers. Shared citations alone are not duplication.
2. For each duplicate:
   - **merge** when the two notes have the same main subject: pick the canonical slug (prefer the slug with a general name over one prefixed by a paper name, e.g. `fmlm-`, `lafa-`, `llmmem-`; else the longer, more complete note). Rewrite the canonical file to contain the union of facts from both, organised as one note. Paper-specific facts stay, attributed to the paper (e.g. 「在 No.13（Learning Unmasking Policies…）中…」). Merge frontmatter: `arxiv` = union, `sources` = union, `related` = union minus the slugs being removed, `tags` = union. The 「屬於[總覽](hub.html)」 line lists every hub the merged notes belonged to. Leave the removed file in place; the coordinator deletes it.
   - **trim** when only a section overlaps: cut the re-explanation from the note whose main subject it is not, and replace it with one or two sentences plus a link to the note that owns the topic. Do not lose facts that exist only in the trimmed passage; move them into the owning note.
3. Edit only the files assigned to you (hubs listed as read-only you may read but not edit). Links in other files that point to removed slugs are rewritten by the coordinator, including `related:` lists.
4. Write {"merged": {"<removed-slug>": "<canonical-slug>", ...}, "trimmed": ["<slug>", ...]} to the mapping file. Use an empty object or list when there is nothing.

Return at most 10 lines: each merge as `removed → canonical (one-line reason)`, each trim as `slug: what was cut, now links to X`.
