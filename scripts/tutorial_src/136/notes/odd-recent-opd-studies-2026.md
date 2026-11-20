---
slug: odd-recent-opd-studies-2026
title: 2026 年的 on-policy 蒸餾研究：演算法變體、診斷與失敗模式
cluster: RL, Reasoning & Self-Distillation
kind: topic
tags: [on-policy-distillation, survey, failure-modes, OPD]
arxiv: ["2609.35259", "2604.00626", "2604.13016", "2605.10889", "2605.06387", "2606.02684", "2605.11182", "2606.13657"]
sources: ["2609.35259 §2", "arXiv abstract 2604.00626", "arXiv abstract 2604.13016", "arXiv abstract 2605.10889", "arXiv abstract 2605.06387", "arXiv abstract 2606.02684", "arXiv abstract 2605.11182", "arXiv abstract 2606.13657"]
related: ["odd-lineage-hub", "odd-onpd-vs-offpd-setup", "on-policy-distillation-imitation", "context-distillation-self-distillation", "odd-on-policy-forgetting-sparsity-claims", "odd-onpd-vs-offpd-setup"]
---
# 2026 年的 on-policy 蒸餾研究

屬於 [蒸餾動態研究的研究脈絡總覽](odd-lineage-hub.html)。論文相關工作（arXiv:2609.35259 §2）把這批文獻歸為「OnPD 的演算法變體與失敗模式」，並指出其中很少直接對比 OnPD 與 OffPD。前置概念見 [On-policy 蒸餾與線上模仿學習](on-policy-distillation-imitation.html)。

- Song and Zheng（arXiv:2604.00626）：on-policy 蒸餾的綜述，以學生產生的狀態、教師資訊與策略更新的互動建立統一視角。
- Li et al.（arXiv:2604.13016）：系統性研究 OPD 動態，摘要指出成功需要師生思考模式相容，且教師須提供學生未見過的新能力。本論文因此在每個任務上另訓練教師，以確保能力差距。
- Armandpour et al.（arXiv:2605.10889）：摘要提出免訓練的診斷框架，在每個 token、問題與教師的解析度上判斷 OPD 訊號何時有益、何時有害。
- Jia et al.（arXiv:2605.06387）：摘要指出標準的優勢加權策略梯度 OPD 有高變異、零優勢區梯度消失與探索瓶頸，提出 AOPD。
- Li et al.（arXiv:2606.02684）：FiRe-OPD，先過濾軌跡，再對保留軌跡內的 token 軟加權。
- Zhu et al.（arXiv:2605.11182）：摘要指出數學推理的 OPD 對教師選擇與損失形式敏感，OPSD 在缺少實例特定特權資訊時失敗。自蒸餾背景見 [context distillation 與自蒸餾](context-distillation-self-distillation.html)。
- Yu et al.（arXiv:2606.13657）：OPD 更新的稀疏性與幾何（見 [On-policy 學習的遺忘與更新稀疏說法](odd-on-policy-forgetting-sparsity-claims.html)）。

Lu and Thinking Machines Lab（2025）的部落格文章為論文所引，非 arXiv 論文，未查證。

OnPD 與 OffPD 的定義見 [OnPD 與 OffPD 的定義](odd-onpd-vs-offpd-setup.html)。
