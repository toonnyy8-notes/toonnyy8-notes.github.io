---
slug: mk-judge-biases-and-adversarial-attacks
title: LLM 評審的偏誤與對抗攻擊（本文之前的弱點研究）
cluster: RL, Reasoning & Self-Distillation
kind: topic
tags: [LLM-as-a-judge, judge-bias, adversarial-attack, reward-hacking]
arxiv: ["2507.08794", "2305.17926", "2402.14016", "2410.07137", "2402.10669", "2410.02736", "2406.12624", "2504.03846", "2502.01534", "2504.09946", "2505.22203"]
sources: ["arXiv:2507.08794 附錄 A（Vulnerabilities of LLM-as-a-judge）", "arXiv 摘要 2305.17926、2402.14016、2410.07137、2504.09946、2505.22203"]
related: ["mk-verifiers-and-reference-based-benchmarks", "mk-master-key-attack-and-master-rm", "mk-lineage-hub"]
---
# LLM 評審的偏誤與對抗攻擊

本文附錄 A 把先前工作歸為「偏好比較情境下的偏誤與弱點」，列出 Wang 等 2023、Ye 等 2024、Raina 等 2024、Zheng 等 2024、Chen 等 2024、Huang 等 2025b、Thakur 等 2024、Chen 等 2025、Li 等 2025a、Wang 等 2025。以下只展開本文附錄明確描述、且已在 arXiv 核對的部分。

- **位置偏誤。** Wang 等 2023（arXiv 2305.17926，Large Language Models are not Fair Evaluators）摘要：只改變候選回應在上下文中的出現順序，就能改變品質排名；例如 ChatGPT 作評審時 Vicuna-13B 在 80 題中的 66 題勝過 ChatGPT。
- **通用對抗短語。** Raina 等 2024（arXiv 2402.14016）摘要：在被評文字後串接簡短的通用對抗短語，可讓評審 LLM 預測偏高的分數；絕對評分比比較式評估更容易受攻擊。本文附錄 A 的描述是對低品質回應附加簡單短語會明顯提高被偏好的機率。
- **無關的固定輸出。** 本文附錄 A 引用 Zheng 等 2024：產生無意義字串的模型仍能在多個 LLM-as-a-judge 基準得高分。對應論文推定為 arXiv 2410.07137（Cheating Automatic LLM Benchmarks: Null Models Achieve High Win Rates；本文參考文獻表未列此條，對應關係由標題與摘要比對得出）。其摘要：總是輸出與指令無關之固定回應的 null model，在 AlpacaEval 2.0 達 86.5% LC 勝率。
- **推理模型的偏誤。** Wang 等 2025（arXiv 2504.09946）摘要：LRM 仍受 bandwagon、authority、position、distraction 偏誤影響。本文附錄 A 引述其中在兩個候選回應間插入「wait, let me think about it」會提高對後者的偏好。
- **其他偏誤研究。** Chen 等 2024（2402.10669）、Ye 等 2024（2410.02736）、Thakur 等 2024（2406.12624）、Chen 等 2025（2504.03846，自我偏好）、Li 等 2025a（2502.01534，preference leakage）本文僅列入引用，未逐項描述，此處不補充其內容。

**與本文的差異（附錄 A）。** 上述多為偏好比較；本文處理的是評審拿候選答案與參考答案比對的情境。同期工作 Huang 等 2025c（arXiv 2505.22203）已指出 LLM 獎勵模型在數學推理中會被空符號或無意義回應騙成假陽性；本文的差異是同時評估非字詞符號與新類別 reasoning openers、涵蓋一般推理領域與大型模型，並提出資料擴增緩解。同期工作見 [驗證器與參考式基準](mk-verifiers-and-reference-based-benchmarks.html)。

下一步見 [Master key 攻擊與 Master-RM](mk-master-key-attack-and-master-rm.html)；回到 [總覽](mk-lineage-hub.html)。
