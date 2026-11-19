# AGENTS.md — Paper Master

## 設計意圖

Paper Master 把「研究手記」Discord 伺服器裡收集的論文，整理成一個可查詢、可重複使用的知識庫。目標讀者是兩位作者自己與初學者：先在關聯圖裡找到論文與它的鄰居，再讀對應的研究筆記和互動教學。

三個原則貫穿所有產出：

- **單一事實來源**：每篇論文在 `papers.csv` / `papers.json` 只有一列，每個概念在研究筆記裡只寫一次。
- **高內聚、低耦合**：一篇筆記只談一個主題，其他篇用連結引用，不重複內容。詳細規則見下方「研究筆記」一節。
- **可重新產生**：所有產出都由 `scripts/` 從來源清單建出，改資料就改來源或腳本，不手改產出檔。

## 來源與產出

| 來源 | 分支 |
| --- | --- |
| `ai-research-topic-papers.md`、`diffusion-channels-papers.md` | `raymin-notes` |
| `waiting-to-read-papers.md` | `toonnyy8-notes` |

只收各檔「一、論文」的部分；「二、未收錄內容」裡的 Claude artifact 是教學素材來源。

產出都在 `paper-master` 分支，做完 commit 並 push 到這個分支：

- `papers.csv`（UTF-8 BOM，Excel 直接開）與 `papers.json`
- `index.html`：互動式論文關聯圖
- `tutorials/`：互動教學與研究筆記；`tutorials/index.html` 是總目錄

重新產生的指令與目前的整理方式見 `PAPER-MASTER.md`。互動教學頁的清單在 `scripts/tutorial_list.json`，原始頁面在 `scripts/tutorials_raw/`，每篇教學的設計文件、審查與驗算在 `scripts/tutorial_src/<論文編號>/`。

任何對外可見的變更（新論文、新教學或筆記、介面更新、新文件）都在 `scripts/news.json` 加一則消息（`python3 scripts/news.py add …`），`news.html` 由它產生；`add_tutorial.py` 會自動加教學上架的消息。

## 步驟 1–4：資料庫與關聯圖

1. **讀取**上面三份清單，依 arXiv ID 或網址合併重複論文，保留所有來源。
2. **資料庫欄位**：Paper ID、Title、Authors、Year、URL、Source File & Branch、Source Category、Cluster、Core Topic / Summary、Tags、Related Paper IDs。作者、年份、摘要從 arXiv 補；查不到就留空，不猜。
3. **Tags 與群集**：Tags 要標準化（同義詞合併成一個，例如 Diffusion Model → Diffusion）；群集依 Tags 共現與原始分類歸納，每個群集要有核心主題名稱。
4. **關聯圖**（參考 Connected Papers）：節點＝論文，大小＝關聯度，顏色＝群集；連線＝主題相似，越像距離越近。要有搜尋（高亮直接關聯的論文）、依 Tag／群集／來源分支篩選，以及點節點後顯示完整資訊與最相關論文的側邊欄。

## 步驟 6：每篇論文的詳細介紹

建構互動式教學時使用 `.claude/skills/paper-tutorial/`（Elephant-Goldfish 流程、模型分工、驗收與上架腳本）。以下是內容要求。

依需求逐篇做，三個階段依序完成：

1. **介紹**：簡潔總結研究動機、貢獻與理論，再從作者角度逐步剖析設計概念、理論如何一層層堆疊、實作要注意的細節。
2. **Research 整理資料**：整理相關研究的發展歷程，讓讀者能循序漸進理解本研究。產出寫成研究筆記，遵守下方「研究筆記」的規則。
3. **建構 blog**：依整理好的資料為初學者講解。以 [Anthropic 官網](https://www.anthropic.com/) + [Material Design 3](https://m3.material.io/) 風格為基礎，用 JS、CSS、SVG 做互動式教學、圖片與動畫。能用字面意思就用字面意思，不用比喻或華麗辭藻。視覺以 Awwwards／Webby／FWA 得獎作品為標竿，完成後逐項自我檢查（字體、留白、視覺層級、色彩、動態、微互動、響應式、原創性），有差距就修正再檢查，直到沒有可見的改進空間。

已經有 Claude 做好的互動教學 html 時，直接下載並改成本站格式（加上回到關聯圖與總目錄的導覽、連到對應論文），不重做。

## 研究筆記（高內聚、低耦合）

步驟 6-2 的 research 一律拆成多篇、分群整理，提高復用量並去除冗餘：

- **一篇一主題**：每篇只涵蓋一個方法、一條研究脈絡或一個跨主題的概念，能獨立閱讀與被引用。
- **合併重複**：同一內容出現在多處時只寫成一篇，其他篇用 `[標題](slug.html)` 連結過去。寫新筆記前先檢查 `scripts/notes_split/` 裡有沒有現成的主題可以引用或擴充。
- **總覽（hub）**：每個主題區一篇短總覽，用幾段話說明全貌，並依閱讀順序連到子筆記。
- **保留事實**：數字、arXiv ID、引用照原文，不自行補充或「更新」。
- **frontmatter**：每篇以 YAML 開頭，欄位為 `slug`、`title`、`cluster`（資料庫的 10 個群集之一）、`kind`（`hub` 或 `topic`）、`tags`、`arxiv`（文中引用的所有 arXiv ID）、`sources`、`related`（連到的其他 slug）。`index.json` 依閱讀順序列出所有筆記。

建置時，`arxiv` 欄位決定筆記連到資料庫中的哪些論文，`related` 決定「相關筆記」與總目錄的分組，所以兩欄都要完整。
