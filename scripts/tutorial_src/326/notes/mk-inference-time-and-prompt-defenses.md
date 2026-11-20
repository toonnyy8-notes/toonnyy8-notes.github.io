---
slug: mk-inference-time-and-prompt-defenses
title: 推論時策略與提示設計對假陽性率的影響
cluster: RL, Reasoning & Self-Distillation
kind: topic
tags: [chain-of-thought, majority-voting, prompt-design, scaling, false-positive-rate]
arxiv: ["2507.08794", "2408.15240"]
sources: ["arXiv:2507.08794 §3.4、Table 3、Figure 4"]
related: ["genrm-inference-time-compute-majority-voting", "mk-master-key-attack-and-master-rm", "mk-lineage-hub"]
---
# 推論時策略與提示設計對假陽性率的影響

**背景。** Zhang 等 2024a（Generative Verifiers）指出 CoT 與多數決可提升無參考答案設定下的生成式獎勵模型，見 [推論時運算與多數決](genrm-inference-time-compute-majority-voting.html)。本文 §3.4 在有參考答案的設定檢驗這些做法。

**模型規模（Figure 4）。** 以 Qwen2.5-Instruct 0.5B 至 72B 測試，FPR 呈非單調：0.5B 的 FPR 最低但與 GPT-4o 的一致性最弱；1.5–3B FPR 急升而一致性提高；7–14B 兼顧低 FPR 與高一致性；32B、72B 的 FPR 再度上升。作者說機制尚未釐清，初步假說在附錄 C。

**CoT 與多數決（Table 3a）。** 使用 CoT 提示加五次生成多數決，在部分設定的 FPR 反而較高：Qwen2.5-72B 整體平均 | 最差由 No-CoT 的 66.8 | 90.9 變為 CoT 的 50.9 | 97.0；Qwen2.5-7B 由 12.6 | 31.0 變為 40.4 | 91.3。本文結論是這些技術並非在所有情況都可靠，應謹慎使用。

**移除問題（Table 3b）。** 評估提示只含回應與參考答案時 FPR 明顯下降，例如 Qwen2.5-72B 在 Multi-subject RLVR 由 51.4 | 70.5 降到 4.9 | 10.8。本文建議數學任務可省略問題，但一般推理常需要問題才能判斷。

**自動發現新 master key（§3.4）。** 文字嵌入與 master keys 相近的文字也會引發高 FPR，例如「mental process」使 GPT-4o 在 GSM8K 的 FPR 為 16.1%。

回到 [Master key 攻擊與 Master-RM](mk-master-key-attack-and-master-rm.html) 或 [總覽](mk-lineage-hub.html)。
