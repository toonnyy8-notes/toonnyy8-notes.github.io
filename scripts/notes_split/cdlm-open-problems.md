---
slug: cdlm-open-problems
title: 連續擴散語言模型的未解決問題與後續方向
cluster: Diffusion Language Models
kind: topic
tags: [open-problems, continuous-diffusion-lm, limitations]
sources: [flow-matching-lm-2026.md]
arxiv: ["2610.02665", "2605.11125", "2606.27617", "2607.00588", "2610.13834", "2609.04531"]
related: ["continuous-flow-lm-overview", "flow-map-few-step-distillation", "self-conditioning-flow-lm", "cdlm-representation-space", "cdlm-likelihood-estimation", "hybrid-continuous-discrete-lm", "cdlm-scaling"]
---
# 連續擴散語言模型的未解決問題與後續方向

屬於 [2026 年連續 flow 語言模型總覽](continuous-flow-lm-overview.html)。各論文自述的限制：

1. **大模型少步品質。**（見 [少步蒸餾](flow-map-few-step-distillation.html)）
   - Sigma 自述在低步數下（DDIM-4/8）表現大幅下降，高步數時每 token 解碼仍比最佳化過的 AR 慢。 [arxiv](https://arxiv.org/pdf/2610.02665v1)
   - Scaling CFM 自述蒸餾後的 Gen PPL 仍高，ESD 在大規模下不穩定。
2. **對 AR 的差距。**
   - RePlaid 的算力倍數約 20×。
   - Scaling CFM 的 1.7B 模型 Gen PPL 為 58.66，AR 為 45.33。
   - S-FLM 自述「a substantial gap to autoregressive decoding remains」。 [arxiv](https://arxiv.org/pdf/2605.11125v3)
   - MLFM 在 GSM8K 上為 31.24，LLaMA-2 為 58.6。 [arxiv](https://arxiv.org/pdf/2606.27617)
3. **低 entropy 與重複。**（見 [self-conditioning](self-conditioning-flow-lm.html)）
   - LangFlow 樣本的 entropy 偏低。
   - ELF 的低 Gen PPL 主要來自重複（2607.00588）。 [arxiv](https://arxiv.org/html/2607.00588v1)
   - JSD 與 forward KL 蒸餾在 entropy 與品質之間有取捨（Scaling CFM）。
4. **one-hot 的記憶體與計算成本。** 成本隨 |V| 與 L 線性增加（Scaling CFM、S-FLM）。超球面或低維 embedding 能避開這個問題，但又必須處理 embedding collapse：Sigma 未正則化時塌縮到約 4 維（見 [表示空間](cdlm-representation-space.html)）。 [arxiv](https://arxiv.org/pdf/2610.02665v1)
5. **likelihood 評估。** 半離散 ELBO 的緊度未知，CFM 在 MCQA 上低於以 ELBO 訓練的離散擴散（見 [likelihood 估計](cdlm-likelihood-estimation.html)）。S-FLM 的截斷界假設 embedding 隨機分佈，作者稱它只是啟發式。 [arxiv](https://arxiv.org/pdf/2605.11125v3)
6. **超參數遷移。** 時間條件層不在 µP / Complete(d)P 的涵蓋範圍內（Scaling CFM）。
7. **混合方法與連續工具不相容。** SemiCat 的分段軌跡難以套用連續加速工具。 [arxiv](https://arxiv.org/html/2610.13834) FMLM+ 的 PR 固定已提交的 token，作者寫明無法回頭修正（見 [混合方法](hybrid-continuous-discrete-lm.html)）。
8. **實驗規模與重複次數。** S-FLM 每個設定只訓練一次（單次 TinyGSM 訓練超過 300 美元）。 [arxiv](https://arxiv.org/pdf/2605.11125v3) FMLM+ 自述實驗規模小、資料多為合成。
9. **研究誠信面。** PlaidQ（2609.04531）已撤稿； [alphaxiv](https://www.alphaxiv.org/abs/2609.04531) Lafa 的部分彙總數字目前只見於二手報導。 [aiweekly](https://aiweekly.co/alerts/lafa-flow-language-model-hits-254x-ar-throughput-at-7b-scale)

## 後續值得追蹤的方向
- 解決大模型少步品質下降的方法（Sigma 的 SUSReg / PDD、DMD for CDLMs）。
- 不需實體化 one-hot 的 CE 參數化（S-FLM、ConvergeFlow 的凸包預測器）。
- Lafa 原文表格的逐項數字，以及它以 flow 草稿加 AR 驗證的吞吐量優勢能否在其他實作中重現（見 [規模化策略](cdlm-scaling.html)）。
