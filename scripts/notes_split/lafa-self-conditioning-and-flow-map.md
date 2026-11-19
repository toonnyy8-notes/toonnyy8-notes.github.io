---
slug: lafa-self-conditioning-and-flow-map
title: self-conditioning 與 flow map 蒸餾在 Lafa 的使用
cluster: Diffusion Language Models
kind: topic
tags: [self-conditioning, flow-map, distillation, few-step]
arxiv: ["2610.28559", "2208.04202", "2505.18825", "2602.16813", "2607.00714", "2207.12598"]
sources: ["arXiv abstracts", "paper.txt §2.2, §2.3"]
related: ["lafa-lineage-overview", "lafa-embedding-flow-denoiser", "self-conditioning-flow-lm", "flow-map-few-step-distillation", "fmlm-flow-map-distillation-lineage"]
---
# self-conditioning 與 flow map 蒸餾

屬於 [Lafa 研究脈絡總覽](lafa-lineage-overview.html)。

## self-conditioning
- Analog Bits（Chen et al., 2022，2208.04202）提出以模型自己前一步的估計作為條件輸入，論文以此作為出處 [37]。
- Lafa 訓練時先做一次無梯度的去噪預測，再把它作為輸入做第二次前向並計算損失；推論時直接用上一步的預測，不增加前向次數（paper.txt §2.2）。
- Self-conditioned Flow Map Language Models via Fixed-point Flows（2607.00714）為論文引用的相關理論，摘要稱其解釋 self-conditioning 的效果。完整討論見 [self-conditioning 在連續 flow 語言模型中的作用](self-conditioning-flow-lm.html)。

## flow map 蒸餾
- Flow map X_{s,t} 把狀態從時間 s 直接傳送到 t。How to Build a Consistency Model（Boffi et al.，2505.18825）的摘要把 flow map 模型與 consistency model 視為同一類，並提出自蒸餾的學習方法。
- Flow Map Language Models（2602.16813）提出單純形（simplex）值的兩時間去噪器；Lafa 把它擴充到非同步 flow，並讓學生以教師的對角預測與兩段組合的 stop-gradient 目標蒸餾，使生成最少可降到一步（paper.txt §2.2）。
- 發展歷程見 [Flow map 與少步蒸餾的發展](fmlm-flow-map-distillation-lineage.html) 與 [連續 flow 語言模型的少步生成](flow-map-few-step-distillation.html)。

## 推論時的引導
Lafa 以 prefix guidance 加強前綴條件，類似 classifier-free guidance（Ho and Salimans，2207.12598），每步多一次不含前綴的評估（paper.txt §2.3）。
