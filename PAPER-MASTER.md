# Paper Master

互動式論文關聯圖：<index.html>（GitHub Pages 開啟或本機直接打開皆可）

- 資料來源：`raymin-notes` 分支的 `ai-research-topic-papers.md`、`diffusion-channels-papers.md`，以及 `toonnyy8-notes` 分支的 `waiting-to-read-papers.md`、`pdf-library-papers.md`（只收各檔「一、論文」部分）
- `papers.csv`：Excel 可直接開啟（UTF-8 BOM）。欄位：Paper ID、Title、Authors、Year、URL、Source File & Branch、Source Category、Cluster、Core Topic / Summary、Tags、Related Paper IDs
- `papers.jsonl`：同一份資料，一篇論文一行（JSON Lines），給關聯圖與重建使用；改動時 git diff 只顯示變動的論文
- `papers.db`：同一份資料的 SQLite 版，多值欄位拆成獨立的表：`papers`、`authors`、`tags`、`sources`、`categories`、`related`。例：
  ```sh
  sqlite3 papers.db "SELECT p.id, p.title FROM papers p JOIN tags t ON t.paper_id = p.id WHERE t.tag = 'Diffusion' AND p.year >= '2025' ORDER BY p.degree DESC LIMIT 10"
  ```
- `tutorials/`：10 份 Claude 互動教學（加上回到關聯圖的導覽列），以及研究筆記：6 份原始長筆記（`scripts/notes_raw/`）依「高內聚、低耦合」拆成 6 篇總覽＋44 篇主題筆記（`scripts/notes_split/`），每篇連到資料庫中引用的論文
- `news.html`：最新消息（更新公告、新論文、互動教學、文件），資料在 `scripts/news.json`；新增一則：`python3 scripts/news.py add <update|paper|tutorial|note|doc> "標題" "內文" [連結文字=網址 ...]`
- `AGENTS.md`：專案設計意圖與各步驟的整理規範

## 整理方式

- 重複論文依 arXiv ID 或網址合併，保留所有來源
- Paper ID 永久不變：重建時沿用前一版 `papers.jsonl` 的編號，新論文接在最大編號之後（教學與筆記以 `#paper-<id>` 連結論文）
- 作者、年份、摘要由 arXiv API 補上；非 arXiv 連結這三欄留空
- Tags：以關鍵字規則從標題、原分類（與摘要）擷取並標準化成 37 個標籤
- Clusters：12 個研究群集，依原始分類（人工整理的討論串／頻道）對應決定
- Related Paper IDs：Tag 加權 Jaccard 相似度＋同原分類加分，取前 6 篇

## 重新產生

```sh
python3 scripts/tutorials.py && python3 scripts/render.py && python3 scripts/news.py   # 從 scripts/ 的資料重建 tutorials/、index.html、news.html
```

資料庫本身（papers.csv／papers.jsonl／papers.db）由 `scripts/parse.py` 與 `scripts/build.py` 產生，需先把四份清單 md 放在 `scripts/`。新增互動教學用 `.claude/skills/paper-tutorial/`。

## 尚未完成

- 其餘論文的互動教學頁依需求再做
