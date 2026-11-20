---
slug: r1-process-supervision-prm
title: 過程監督與 PRM：結果與步驟回饋的比較
cluster: RL, Reasoning & Self-Distillation
kind: topic
tags: [PRM, process-supervision, outcome-supervision, Math-Shepherd, reward-hacking]
arxiv: ["2211.14275", "2305.20050", "2312.08935", "2210.10760", "2408.03314", "2501.12948"]
sources: ["arXiv 摘要 2211.14275、2305.20050、2312.08935、2210.10760、2408.03314", "2501.12948 附錄 G.2、H.2、H.3、§2.2"]
related: ["r1-lineage-hub", "r1-rlhf-ppo-background", "r1-grpo-outcome-rewards", "rlvr-grpo-credit-assignment"]
---
# 過程監督與 PRM：結果與步驟回饋的比較

**問題設定。** 推理任務的回饋可以只看最終結果（outcome supervision），也可以對每個中間步驟給回饋（process supervision）。
- Uesato 等（2022，arXiv 2211.14275）摘要：提出此問題，比較結果式與過程式監督。
- Lightman 等（2023，arXiv 2305.20050）摘要：同樣比較兩種監督，並強調人類回饋成本高；其訓練的 process reward model 見 [RLVR、GRPO 與信用分配瓶頸](rlvr-grpo-credit-assignment.html)。
- Wang 等（2023，arXiv 2312.08935，Math-Shepherd）摘要：以自動建構的過程監督資料訓練對每一步給分的 PRM，不依賴人工標註，用於驗證（重排序）與強化學習。

**DeepSeek-R1 的嘗試與結論（G.2）。** 本文表示 PRM 能引導模型走向較好的解法，但實驗中有三項限制：
1. 一般推理中難以明確定義細粒度的「步驟」。
2. 判斷中間步驟是否正確困難；模型自動標註效果不佳，人工標註難以擴充。
3. 引入模型式 PRM 會導致 reward hacking（引用 Gao 等 2022，見 [LLM 的 RL 前史](r1-rlhf-ppo-background.html)），重新訓練獎勵模型需額外資源並使流程複雜。

本文結論：PRM 能重排序 top-N 回應或輔助搜尋（引用 Snell 等 2024），但在大規模 RL 中，優點不抵額外運算成本。因此 §2.2 的獎勵設計不使用神經獎勵模型（見 [GRPO 與結果導向獎勵](r1-grpo-outcome-rewards.html)）。

本文附註：這些失敗不代表此類方法無法建立有效的推理模型。總覽：[總覽](r1-lineage-hub.html)。
