---
slug: lopd-latent-memory-for-agents
title: Latent memory：把經驗壓縮成模型可讀的連續 token
cluster: RL, Reasoning & Self-Distillation
kind: topic
tags: [latent-memory, agent-memory, experience, compression]
arxiv: ["2608.13040", "2402.04624", "2510.00444", "2509.24704", "2601.05505", "2605.30690"]
sources: ["arXiv abstract 2402.04624", "arXiv abstract 2510.00444", "arXiv abstract 2509.24704", "arXiv abstract 2601.05505", "arXiv title 2605.30690", "2608.13040 §2 Related Work"]
related: ["retrieval-augmented-external-memory", "lopd-latent-reasoning-continuous-tokens", "lopd-method-learnable-context-and-margin"]
---
# Latent memory：把經驗壓縮成模型可讀的連續 token

LOPD 論文第 2 節把 latent state 視為程序性、事實性與經驗性知識的緊湊載體，避免暴露冗長文字。外部檢索式記憶見 [檢索增強與外部記憶](retrieval-augmented-external-memory.html)。

## 脈絡
- Wang et al.（2024, arXiv:2402.04624）MemoryLLM：摘要描述在 transformer latent 空間中放一個固定大小的記憶池，可自我更新。
- Wu et al.（2025, arXiv:2510.00444）TokMem：摘要描述把每個可重用的任務程序編譯成一個可訓練的記憶 token，backbone 凍結。
- Zhang, Fu, Yan（2025, arXiv:2509.24704）MemGen：摘要描述 memory trigger 監看推理狀態，memory weaver 產生 latent token 序列作為機器原生記憶。
- Hou et al.（2026, arXiv:2601.05505）FlashMem：摘要描述從推理中的 hidden state 蒸餾內在記憶，並重用 backbone 的 KV cache。
- Feng et al.（2026, arXiv:2605.30690）ElasticMem：LOPD 論文列為經驗性 latent memory 的例子，本筆記只核對標題。

## 與 LOPD 的關係
LOPD 論文說明它使用同類 latent 記憶，但角色不同：latent 狀態是 on-policy 蒸餾中老師的可學習特權脈絡，學生在推論時不觀察它。LOPD 的 composer 以 QFormer 風格的交叉注意力把每筆檢索到的經驗壓成 K 個 latent token，見 [LOPD 方法](lopd-method-learnable-context-and-margin.html)。
同為 latent 形式的另一條來源見 [Latent reasoning](lopd-latent-reasoning-continuous-tokens.html)。
