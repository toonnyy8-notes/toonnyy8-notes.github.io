---
slug: inverse-dynamics-state-diffusion
title: 只擴散狀態序列，以逆動力學模型取出動作
cluster: RL, Reasoning & Self-Distillation
kind: topic
tags: [decision-diffuser, inverse-dynamics, state-diffusion, receding-horizon, temporal-unet]
arxiv: ["2211.15657", "1606.07419", "1804.08606", "2205.09991", "2006.11239"]
sources: ["paper.txt §3.1, §3.4, Algorithm 1, Appendix F", "1606.07419 與 1804.08606 僅依本論文的引用方式描述", "arXiv API metadata"]
related: ["decision-diffuser-lineage-hub", "trajectory-diffusion-planning", "ddpm-training-and-sampling-recap", "classifier-free-guidance-return-conditioning", "dd-experiments-and-ablations", "dd-limitations-stochastic-dynamics"]
---
# 只擴散狀態序列，以逆動力學模型取出動作

屬於[總覽](decision-diffuser-lineage-hub.html)。

## 設計
[Diffuser](trajectory-diffusion-planning.html) 對狀態與動作合併的序列做擴散。本論文（arXiv 2211.15657）只對狀態序列做擴散：

x_k(τ) := (s_t, s_{t+1}, ..., s_{t+H-1})_k

其中 k 為前向過程的步數，H 為序列長度。動作由逆動力學模型 f_φ 由相鄰兩個狀態估計：a_t := f_φ(s_t, s_{t+1})。

論文說明不直接擴散動作的理由：RL 中狀態通常連續，動作型態較多樣、常為離散；以關節力矩表示的動作序列頻率較高、較不平滑，較難預測與建模。

## 逆動力學的來源
論文引用 Agrawal 等（2016）與 Pathak 等（2018）作為逆動力學模型的出處。經 arXiv API 核對：
- Agrawal 等，"Learning to Poke by Poking: Experiential Learning of Intuitive Physics"，arXiv 1606.07419（2016）。
- Pathak 等，"Zero-Shot Visual Imitation"，arXiv 1804.08606（2018）。

兩篇的內容僅依本論文的引用，未另外摘錄摘要。

## 訓練與推論
- 同一份離線資料同時訓練反向擴散過程（噪聲模型 ε_θ）與逆動力學 f_φ，損失為兩項相加：噪聲預測的平方誤差，以及 ||a − f_φ(s, s')||²。
- ε_θ 使用 temporal U-Net（沿用 Janner 等 2022 的 Diffuser 架構），f_φ 為 MLP。逆動力學以單一轉移 (s, a, s') 訓練，不需要整條軌跡。
- 推論時為 receding-horizon 迴圈（Algorithm 1）：觀察狀態、在歷史 C 個狀態的約束下取樣未來狀態序列、由 x_0(τ) 取 (s_t, s_{t+1})、執行 a_t = f_φ(s_t, s_{t+1})，再重複。

## 實驗證據
- 表 2（Hopper）：DD 在 Med-Expert、Medium、Med-Replay 分別為 111.8、79.3、100；CondDiffuser（同時擴散狀態與動作，使用無分類器引導）分別為 111.3、66.3、76.5。論文結論是學習逆動力學是擴散動作的一個好替代。
- 附錄 F（Block Push）：位置控制的動作軌跡平滑，DD 與 CondDiffuser 表現相近；力矩控制的動作軌跡含較高頻成分，CondDiffuser 較差。這與論文的動機一致。
