---
slug: training-inference-mismatch-sources
title: 訓練與推論不一致（TIM）的來源：精度、kernel 與 staleness
cluster: RL, Reasoning & Self-Distillation
kind: topic
tags: [training-inference-mismatch, quantization, fp16, staleness, asynchronous-rl]
arxiv: ["2609.20807", "2510.26788", "2601.18150", "2601.14243", "2510.11370", "2505.24298", "2509.19128", "2409.19256"]
sources: ["arXiv:2609.20807 §2.1, §5", "arXiv abstract 2510.26788", "arXiv abstract 2505.24298", "arXiv abstract 2509.19128", "arXiv abstract 2603.19470"]
related: ["importance-sampling-corrections-llm-rl", "score-centering-drift-decomposition"]
---
# 訓練與推論不一致（TIM）的來源

在理論上 trainer 與 sampler 是同一個模型，輸出應相同。本文（§2.1）把 TIM 的來源依嚴重程度整理如下。

1. **兩套引擎的精度或程式碼不同**：不同精度或獨立的程式庫，可能有難以察覺的差異（Liu et al., 2025a；FlashInfer contributors, 2026）。
2. **浮點運算的非結合性**：相同 kernel 在不同輸入形狀下改變歸約順序。Batch-invariant kernel 可消除此依賴，但需要大量工程且 GPU 使用率較低（He & Thinking Machines Lab, 2025）。Qi et al.（arXiv 2510.26788）在摘要中指出 BF16 的捨入誤差是不一致的根源，改用 FP16 即可消除；本文補充 FP16 不能完全消除。
3. **更新 staleness**：為提高 GPU 使用率而分離訓練與推論並非同步執行。AReaL（Fu et al., arXiv 2505.24298）在摘要中描述其完全非同步的 RL 系統；PipelineRL（Piché et al., arXiv 2509.19128）以 in-flight weight update 在硬體效率與 on-policy 程度間取捨。本文指出同一批資料、甚至同一條 rollout 的不同子序列可能來自不同 checkpoint。
4. **低精度 rollout**：FP8-RL（arXiv 2601.18150）、Jet-RL（arXiv 2601.14243）與 MoE router 對齊（arXiv 2510.11370）為本文引用的相關工作；本文僅描述其主題，未逐一重述結果。

本文（§5.1）指出文獻對 TIM 的嚴重程度看法不一：Qiu et al.（2026）發現 fp8 rollout 僅用 token-level IS 即穩定，Microsoft AI Team（2026）則看到即使盡量減少 TIM 仍發散。本文的解釋是 drift 隨訓練累積，輕微的 TIM 也會造成不穩定，只是需要更多步數。因此實驗刻意放大 TIM，包含高斯權重雜訊、量化 sampler 與每 64 步才更新一次的 staleness。對應的修正方法見 [LLM RL 中的 importance sampling 修正](importance-sampling-corrections-llm-rl.html)，本文方法見 [Score centering：drift 與 signal 的分解](score-centering-drift-decomposition.html)。
