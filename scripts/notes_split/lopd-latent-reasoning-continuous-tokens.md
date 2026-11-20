---
slug: lopd-latent-reasoning-continuous-tokens
title: Latent reasoning：以連續向量取代自然語言 token
cluster: RL, Reasoning & Self-Distillation
kind: topic
tags: [latent-reasoning, continuous-thought, Coconut]
arxiv: ["2608.13040", "2412.06769", "2507.06203", "2510.15522", "2602.08332", "2604.02029"]
sources: ["arXiv abstract 2412.06769", "arXiv title 2507.06203", "arXiv title 2510.15522", "arXiv title 2602.08332", "arXiv title 2604.02029", "2608.13040 §2 Related Work"]
related: ["lopd-latent-memory-for-agents", "lopd-method-learnable-context-and-margin"]
---
# Latent reasoning：以連續向量取代自然語言 token

## 脈絡
- Hao et al.（2024, arXiv:2412.06769）Coconut：摘要描述把 LLM 最後一層 hidden state 作為「連續思考」，不解碼成詞，直接作為下一步輸入 embedding 回饋給模型。
- LOPD 論文第 2 節將 latent reasoning 描述為用連續 latent token 或 hidden state 取代自然語言，以擴大模型內部計算量，並引用 Deng et al.（2026, arXiv:2510.15522，Chain of Superposition）、Amos et al.（2026, arXiv:2602.08332，Latent Reasoning with Supervised Thinking States）。
- 綜述：Zhu et al.（2025, arXiv:2507.06203）A Survey on Latent Reasoning；Yu et al.（2026, arXiv:2604.02029）The Latent Space。本筆記只核對了標題與作者，未讀摘要以外的內容。

## 與 LOPD 的關係
LOPD 論文指出 latent token 在這條線中表示中間計畫或子目標，並可微分。LOPD 借用「可微分的連續 token 作為模型輸入」這個形式，但用途是當老師的特權脈絡：學生不看到這個脈絡。注入方式是以 placeholder 位置替換 embedding（論文附錄 A.4），見 [LOPD 方法](lopd-method-learnable-context-and-margin.html)。另一條來源見 [Latent memory](lopd-latent-memory-for-agents.html)。
