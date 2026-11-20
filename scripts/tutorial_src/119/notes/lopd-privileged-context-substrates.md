---
slug: lopd-privileged-context-substrates
title: 特權脈絡的各種實體：答案、回饋、技能與同儕 rollout
cluster: RL, Reasoning & Self-Distillation
kind: topic
tags: [privileged-context, OPSD, skills, feedback, peer-rollouts]
arxiv: ["2608.13040", "2601.18734", "2601.20802", "2604.12002", "2604.10674", "2605.12652", "2606.11709", "2605.23493", "2604.03128"]
sources: ["2608.13040 §1", "2608.13040 §2 Related Work", "arXiv abstract 2604.12002", "arXiv abstract 2604.10674", "arXiv abstract 2605.12652", "arXiv abstract 2606.11709", "arXiv abstract 2605.23493", "arXiv abstract 2604.03128"]
related: ["lopd-opsd-self-teacher-privileged-context", "rich-feedback-and-retrospection", "lopd-opd-failure-modes-and-stabilization", "lopd-method-learnable-context-and-margin"]
---
# 特權脈絡的各種實體

LOPD 論文第 2 節把 OPSD 依「老師看到什麼」分類。以下對照該分類與各篇的 arXiv 摘要。

| 類別 | 工作 | 摘要中的描述 |
|---|---|---|
| 驗證過的答案與推理軌跡 | OPSD（2601.18734） | 老師條件於特權推理軌跡 |
| 環境回饋 | SDPO（2601.20802） | 老師帶回饋，見 [豐富回饋與事後回顧](rich-feedback-and-retrospection.html) |
| 自我修正訊號 | SD-Zero（2604.12002） | 單一模型分 Generator 與 Reviser，Reviser 條件於初始回答與二元獎勵，再蒸餾回 Generator |
| 技能（自然語言） | Skill-SD（2604.10674） | 完成的軌跡被摘要為技能，只餵給老師 |
| 同儕 rollout | MOPD（2605.12652） | 老師同時條件於同組成功與失敗的 rollout |
| 對比證據 | RLCSD（2606.11709） | 對比正確提示與錯誤提示下的師生差距 |
| 檢索或證據引導 | EDGE-OPD（2605.23493） | 以證據引導處理特權資訊造成的行為偏移 |
| 與 RLVR 結合 | Self-distilled RLVR（2604.03128） | 摘要指出只用特權老師的訊號會資訊洩漏 |

## 與 LOPD 的關係
LOPD 論文認為這些脈絡都是設計者預先決定的離散 artifact（答案、反思文字、成功 rollout、檢索文件）。Skill-SD 摘要自己也指出固定的特權資訊無法涵蓋 agent 任務中多樣的有效策略。LOPD 實驗中的基線 OPSD、SDPO、Skill-SD 分別對應答案或軌跡、回饋、技能三種實體。方法見 [LOPD 方法](lopd-method-learnable-context-and-margin.html)；這些實體引出的穩定性問題見 [OPD 與 OPSD 的失敗模式與穩定化](lopd-opd-failure-modes-and-stabilization.html)。
上游的 OPSD 方法見 [OPSD：同一模型帶特權脈絡當老師](lopd-opsd-self-teacher-privileged-context.html)。
