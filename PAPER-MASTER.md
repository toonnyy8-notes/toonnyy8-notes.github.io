# Paper Master

互動式論文關聯圖：<index.html>（GitHub Pages 開啟或本機直接打開皆可）

- 資料來源：`raymin-notes` 分支的 `ai-research-topic-papers.md`、`diffusion-channels-papers.md`，以及 `toonnyy8-notes` 分支的 `waiting-to-read-papers.md`（只收各檔「一、論文」部分）
- `papers.csv`：Excel 可直接開啟（UTF-8 BOM）。欄位：Paper ID、Title、Authors、Year、URL、Source File & Branch、Source Category、Cluster、Core Topic / Summary、Tags、Related Paper IDs
- `papers.json`：同一份資料，給關聯圖使用
- `tutorials/`：清單中已有的 Claude 互動教學，下載後加上回到關聯圖的導覽列

## 整理方式

- 重複論文依 arXiv ID 或網址合併，保留所有來源
- 作者、年份、摘要由 arXiv API 補上；非 arXiv 連結這三欄留空
- Tags：以關鍵字規則從標題、原分類（與摘要）擷取並標準化成 33 個標籤
- Clusters：10 個研究群集，依原始分類（人工整理的討論串／頻道）對應決定
- Related Paper IDs：Tag 加權 Jaccard 相似度＋同原分類加分，取前 6 篇

## 重新產生

```sh
cd scripts
# 先把三份 md 放在同一個資料夾
python3 parse.py && python3 build.py && python3 tutorials.py && python3 render.py
```

## 尚未完成

- 步驟 6 只處理了已存在的 Claude artifact（10 份）；6 個 `claude.ai/public/artifacts/...` 連結無法用工具下載，其餘論文的教學頁依需求再做
