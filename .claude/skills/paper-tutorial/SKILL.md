---
name: paper-tutorial
description: 為 Paper Master 資料庫中的論文建構互動式教學（AGENTS.md 步驟 6），以 Elephant-Goldfish 流程分派 subagent 並驗收。Use when asked to build, batch-build, or redo an interactive tutorial (互動式教學／互動教學) for a paper in papers.json.
---

# paper-tutorial

為一篇（或一批）論文產出：研究筆記（併入 `scripts/notes_split/`）、教學設計文件、單檔互動教學頁（`tutorials/<slug>.html`）。規範以 `AGENTS.md`（步驟 6、研究筆記）與 `DESIGN.md` 為準；本 skill 只規定流程與分工。

**Elephant**：每篇一個長脈絡 Opus session，持有教學意圖，寫設計文件並建構頁面，整個流程都用 SendMessage 續用同一個 agent。**Goldfish**：無脈絡、單次的審查 session，只讀指定檔案。

## 角色與模型

| 工作 | 模型 | prompt |
| --- | --- | --- |
| 理解備忘（只給 ⭐ 難度論文） | fable | `prompts/advisor.md` |
| 研究脈絡筆記 + timeline.json | sonnet | `prompts/notes.md` |
| 設計文件、修訂、建構、修補（Elephant） | opus | `prompts/design.md`、`prompts/build.md` |
| 忠實度審查、教學與實作審查（Goldfish） | sonnet ×2 | `prompts/review-fidelity.md`、`prompts/review-pedagogy.md` |
| 理解 + 就緒檢查（Goldfish） | haiku | `prompts/readiness.md` |
| 成品驗收（Goldfish） | haiku | `prompts/accept.md` |

prompt 檔中的 `{ID}`（資料庫編號）、`{ARXIV}`、`{TITLE}`、`{SLUG}`、`{DIR}`（`scripts/tutorial_src/{ID}` 的絕對路徑）、`{REPO}`（repo 根目錄絕對路徑）在派工前替換。

## 步驟

1. **選題與準備**
   - 已有 Claude 教學（`scripts/tutorial_list.json` 有對應 paper id）的論文不重做。
   - 批次時依先備關係排序，⭐ 標記推導重、理論密集的論文。
   - 每篇執行 `bash .claude/skills/paper-tutorial/scripts/fetch_paper.sh {ID} {ARXIV}`。它建立 `{DIR}/src/paper.txt`。
   - 完成條件：每篇的 `paper.txt` 存在且非空。
2. **平行派工**（同一則訊息送出）
   - Sonnet 寫筆記。
   - ⭐ 論文先跑 Fable 理解備忘（寫入 `{DIR}/advisor.md`）。
   - Opus Elephant 寫 `{DIR}/design.md`：⭐ 論文等備忘完成再派；其他論文與筆記同時派。
   - 多篇論文的管線互相獨立，同時跑。
   - 完成條件：`design.md` 含學習目標表（題目、參考答案、章節）、互動元件規格、驗收數值表，與 `{DIR}/verify/` 驗算腳本。
3. **Goldfish 審查**
   - 兩隻 Sonnet 同時審 `design.md`。
   - 把兩份報告合併成 `{DIR}/reviews/round1.md` 的編號清單（去重、標 P0/P1/P2），SendMessage 交回 Elephant 修訂。
   - Elephant 修完後跑一隻 Haiku readiness（理解檢查與 go/no-go 合併）。NO-GO 時把阻擋項交回 Elephant，再跑一次。
   - 完成條件：GO。
4. **建構**
   - SendMessage 讓 Elephant 依 `prompts/build.md` 建構 `{DIR}/build/index.html`。
   - 完成條件：`?test` 全數通過、impeccable 0 項，且 Elephant 回報 LO 自檢通過。
5. **成品驗收**
   - 執行 `python3 .claude/skills/paper-tutorial/scripts/extract_page_text.py {DIR}`。它產生 `{DIR}/accept/page.txt`（已移除答案框）與 `questions.txt`。
   - 派 Haiku 驗收，對照 `design.md` 的參考答案逐題評分。
   - 未完全答對的題目交回 Elephant 補正文，修完只重測那幾題。
   - 完成條件：12/12（或全數 LO）能只從頁面答出。
6. **上架**
   - 執行 `python3 .claude/skills/paper-tutorial/scripts/add_tutorial.py {ID} {SLUG} "{TITLE}"`。它併入筆記與 index.json、登錄 `tutorial_list.json`、在最新消息加一則公告，並重建所有頁面。
   - commit 與 push 到 `paper-master`。
   - 完成條件：`tutorials/{SLUG}.html` 存在，且關聯圖詳情卡與總覽筆記都連得到它。

## 回報

只在三個時點回報給使用者：開工（名單與分工）、每篇上架（成品路徑、驗收結果、未解決事項）、遇到阻擋。

## 從試點刪除的冗餘

以 No.238 試點為基準，這些行為已從流程移除：

| 試點做法 | 現行做法 | 原因 |
| --- | --- | --- |
| 理解檢查與就緒檢查各一隻 Haiku | 合併成修訂後的一次 readiness | 第一輪理解檢查的歧義清單與教學審查重疊 |
| 忠實度審查從頭重推所有數值 | 先執行 Elephant 的 `verify/` 腳本，再抽查 | 試點重推花費最多 token，結果與腳本一致 |
| 建構後驗收才發現正文沒寫出答案，多一輪修補 | 建構時要求每個 LO 的參考答案寫進正文，Elephant 自行跑 `extract_page_text.py` 自檢 | 試點 3 題部分正確全因答案只在圖表或答案框中 |
| 每篇重新設計版面，截圖迴圈 3 輪 | 沿用 `tutorials/prediction-targets-k-diff.html` 的版面外殼（tokens、導覽、章節目錄、plate、callout、selfTest），截圖迴圈最多 1 輪，只看新元件與 390px | 版面已通過 award-tier 迴圈與 impeccable |
| 協調者手動抽文字、併筆記、改建置腳本 | `scripts/` 三支輔助腳本 | 減少協調者的 context 與出錯機會 |
| 各階段依序等待、逐階段回報 | 筆記與設計平行、多篇管線平行，只在三個時點回報 | 等待時間大多花在串行 |
