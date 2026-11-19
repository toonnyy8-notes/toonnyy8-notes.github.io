# DESIGN.md

## 概念

研究圖集 + 卡片目錄。關聯圖像星圖：群集名稱直接標在所在區域。每篇論文有目錄編號（No. 273），詳情面板是一張索引卡。

## 參考（BENCHMARK）

| 參考 | 借用的屬性 |
| --- | --- |
| Distill.pub | 襯線大標 + 無襯線內文；680–720px 閱讀欄；文章左側窄欄放章節目錄 |
| Stripe Press | 大量留白；目錄式編號（等寬數字）作為視覺節奏；標題與內文字級比約 4:1 |
| Connected Papers | 圖譜佔滿中央；選取後相關節點高亮、其餘淡出；右側詳情面板 |

## 字型

| 用途 | 字型 | fallback |
| --- | --- | --- |
| 標題（拉丁） | Newsreader 400/500/600 | Georgia, serif |
| 標題（中文） | Noto Serif TC 600/700 | "Songti TC", serif |
| 內文 | IBM Plex Sans 400/500/600 + Noto Sans TC 400/500/700 | system-ui |
| 編號、統計、metadata | JetBrains Mono 400/500 | ui-monospace |

字級（px）：12 / 13 / 15 / 17 / 20 / 24 / 32 / 44 / 64。圖上標籤 13px，最小 12px。

圖中文字大小設定（只在關聯圖頁）：標準 / 大 / 特大，對應 `--fs` = 1 / 1.15 / 1.3，只套用到圖上的論文標題與群集名稱；介面其他文字不變。設定存在 localStorage（`pm-gfs`）；標題的換行與避讓依 `--fs` 重新計算。

## 色彩

| token | light | dark | 用途 |
| --- | --- | --- | --- |
| `--bg` | #F6F4EE | #161513 | 頁面底色 |
| `--surface` | #FFFFFF | #1F1E1B | 面板 |
| `--surface-2` | #EEEBE3 | #2A2925 | 輸入框、hover |
| `--ink` | #1D1C19 | #EEEBE4 | 主文字 |
| `--ink-2` | #57534B | #B5B0A5 | 次要文字 |
| `--ink-3` | #6F6B62 | #8E897F | metadata（AA 對比） |
| `--line` | #DCD7CB | #36342F | 分隔線 |
| `--accent` | #B4532A | #E58C66 | 唯一強調色：選取、連結、focus |
| `--accent-soft` | #F3E2D8 | #3B2820 | 選取背景 |

群集色：10 色、中低彩度，只用在節點與群集圖例，不用在介面元件。

## 間距與形狀

- 間距級距（4px 基準）：4 / 8 / 12 / 16 / 24 / 32 / 48 / 64 / 96。
- 圓角：6（tag）、10（輸入框、按鈕）、16（卡片、面板）、999（pill）。
- 陰影只用在浮動層（圖例、縮放控制、tooltip）。

## 動態

- 時長 160ms（hover）／240ms（面板切換）／400ms（縮放到選取節點）；easing `cubic-bezier(.2,0,0,1)`。
- 每個動畫的功能：hover 回饋、選取狀態改變、縮放時引導視線。
- `prefers-reduced-motion: reduce`：取消縮放動畫與 transition。

## 元件

不使用外部元件庫（應用型介面，但頁面是單檔 HTML + D3，不導入 React/StyleX）；元件依上述 tokens 手寫。

## 審查迴圈紀錄（award-tier-web-design）

截圖寬度 390 / 768 / 1024 / 1440，另含深色模式、鍵盤 focus、reduced-motion 各一張。

| 維度 | 第 1 輪 | 獨立審查 | 第 3 輪後（自評） |
| --- | --- | --- | --- |
| typography | 4 | 4 | 4 |
| whitespace | 4 | 3 | 4 |
| visual_hierarchy | 3 | 3 | 4 |
| color | 4 | 3 | 4 |
| motion | 3 | 3（靜態無法判斷） | 3 |
| micro_interaction | 3 | 3 | 4 |
| responsive | 2 | 2 | 4 |
| originality | 3 | 3 | 3 |
| copy | 4 | 4 | 4 |

| 輪次 | 動詞 | 範圍 |
| --- | --- | --- |
| 1 | adapt、layout | 手機版圖譜移到篩選之前；群集名稱改以群集核心定位，標題避開節點 |
| 2 | layout、colorize、typeset | 群集名稱只放在所屬群集上下方、彼此不重疊；群集色改為 Tableau 10 衍生色；內文字型由 Instrument Sans 換成 IBM Plex Sans（impeccable detect 標記 overused-font） |
| 3 | adapt、layout、animate | 手機版搜尋列置於圖譜上方、其餘篩選收合；選取論文時縮放到整個相關鄰域；圖例不再被縮放控制遮住；關閉的群集淡化並加刪除線；詳情卡切換淡入 |

品質關卡：`npx impeccable detect` 0 項；文字對比皆 ≥ 4.5:1（`--ink-3` 由 #6F6B62 調為 #67635A）；focus 樣式可見；reduced-motion 下取消縮放動畫與 transition。

未解決差距：
- originality 仍為 3：除星圖式群集名稱與目錄編號外，沒有更具辨識度的視覺特徵。
- 總覽狀態下，部分群集名稱仍疊在其他群集的節點上（以底色描邊維持可讀）。
- motion 未以影片驗證，只確認靜態影格與 reduced-motion。
