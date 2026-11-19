---
slug: diffusion-in-rl-concurrent-work
title: 擴散模型在 RL 中的同期與前驅工作：Diffuser、Decision Transformer 與動作層級擴散
cluster: RL, Reasoning & Self-Distillation
kind: topic
tags: [diffuser, decision-transformer, trajectory-generation, model-free, model-based]
arxiv: ["2208.06193", "2205.09991", "2106.01345", "2301.10677", "2211.15657", "2206.11251"]
sources: ["paper.txt §1, §2 Related Work: Diffusion Models in RL", "arXiv API 驗證 2205.09991、2106.01345、2301.10677、2211.15657、2206.11251 的題名與作者；描述依 paper.txt"]
related: ["trajectory-diffusion-planning", "sequence-modeling-offline-rl", "offline-rl-benchmarks-and-baselines", "diffusion-policy-class", "diffusion-ql-loss-coupling", "diffusion-policy-experiments-d4rl"]
---
# 擴散模型在 RL 中的相關工作

論文（arXiv 2208.06193）§1 把離線 RL 的既有做法分為四類：(1) 正則化策略偏離行為策略；(2) 約束價值函數使 OOD 動作得低值；(3) model-based 的悲觀規劃；(4) 以序列預測加回報引導處理。Diffusion-QL 屬第 (1) 類。

## 軌跡層級：Diffuser 與 Decision Transformer

- Diffuser（Janner et al., 2022，arXiv 2205.09991）把整段 state-action 軌跡當作一個擴散樣本，另學回報模型，並在反向取樣時注入其引導。完整說明見 [軌跡層級擴散規劃（Diffuser）](trajectory-diffusion-planning.html)。
- Decision Transformer（Chen et al., 2021，arXiv 2106.01345）以 GPT-2 學軌跡生成器並使用真實軌跡回報；見 [以序列模型處理離線 RL](sequence-modeling-offline-rl.html)。
- 論文指出，序列模型線上使用時無法自回歸預測動作（狀態來自環境），因此每個狀態要預測整段軌跡而只執行第一個動作，計算成本高。

## 動作層級

Diffusion-QL 對動作空間做條件擴散（以狀態為條件），model-free，一次取樣一個動作，Q 值引導在訓練期注入。論文自述 Diffuser 從 model-based 軌跡規劃出發，本方法從 model-free 策略最佳化出發。

論文另引用同樣使用生成模型的決策工作：Pearce et al.（2023，arXiv 2301.10677）以擴散模型模仿人類行為，Ajay et al.（2022，arXiv 2211.15657）論述條件生成模型用於決策，Shafiullah et al.（2022，arXiv 2206.11251）針對多峰行為複製。Diffusion-QL 與 Diffuser 在 D4RL 上的比較數字見 [本論文使用的離線 RL 基準與比較方法](offline-rl-benchmarks-and-baselines.html) 與 [D4RL 實驗與消融](diffusion-policy-experiments-d4rl.html)。
