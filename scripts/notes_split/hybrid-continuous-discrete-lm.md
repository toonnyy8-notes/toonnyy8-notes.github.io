---
slug: hybrid-continuous-discrete-lm
title: 混合連續—離散擴散語言模型
cluster: Diffusion Language Models
kind: topic
tags: [hybrid-diffusion, masked-diffusion, candi, semicat, fmlm-plus]
sources: [flow-matching-lm-2026.md]
arxiv: ["2610.13834", "2606.24773", "2606.27617", "2610.12898"]
related: ["continuous-flow-lm-overview", "cdlm-noise-schedule", "cdlm-open-problems"]
---
# 混合連續—離散擴散語言模型

屬於 [2026 年連續 flow 語言模型總覽](continuous-flow-lm-overview.html)。

- **CANDI（Pynadath et al. 2025）：** 混合連續與離散，提出解碼錯誤率排程，為 FLM 等論文所沿用（見 [噪聲／時間排程](cdlm-noise-schedule.html)）。
- **SemiCat（2610.13834）：**
  - 在 DoubleCat 上加入 CANDI 式的離散 reveal。
  - TinyGSM 達 49.94%（調過取樣器）。 [arxiv](https://arxiv.org/abs/2610.13834)
  - 1.7B 規模下，6 項 likelihood 評測中 4 項勝過離散擴散；無條件生成勝過 one-hot flow。 [arxiv](https://arxiv.org/abs/2610.13834)
  - 作者自述：隨機 reveal 使軌跡只是分段連續，較難套用連續加速工具。 [arxiv](https://arxiv.org/html/2610.13834)
- **FMLM+（2606.24773）：**
  - 在 FMLM 上加入遮罩式噪聲，乾淨位置的 t 固定為 1。
  - 證明 s=t=0 時的目標等同 MDM 目標，因此可從 MDM 初始化或蒸餾。
  - Posterior Refinement 依最後一步 flow map 輸出的 p_max 保留高信心 token，並對其他位置重新加噪。
  - 結果：GSM8K（TinyGSM 訓練）32 NFE 達 19.0%，MDLM 1024 NFE 為 18.0%；從 MDLM 初始化時，64 NFE 達 33.6%；無條件 Sudoku 9 NFE 達 87.1%。
- **MLFM（2606.27617）：** 交替做連續去噪與離散 unmask。CCFG + OTP 取樣器使 GSM8K 由 13.19 升到 31.24。 [arxiv](https://arxiv.org/pdf/2606.27617)
- **Predictive-State Flows（2610.12898）：**
  - 主張 diffusion prior 不應負責細粒度 token 選擇。 [arxiv](https://arxiv.org/abs/2610.12898)
  - 以 rectified flow 生成 K≪L 個 block 層級的 predictive state，由 block decoder 產生 token。 [arxiv](https://arxiv.org/abs/2610.12898) [arxiv](https://arxiv.org/html/2610.12898)

## 限制
- 混合方法與連續工具不相容：SemiCat 的分段軌跡難以套用連續加速工具。FMLM+ 的 PR 固定已提交的 token，作者寫明無法回頭修正。
- FMLM+ 自述實驗規模小、資料多為合成。
- 見 [未解決問題](cdlm-open-problems.html)。
