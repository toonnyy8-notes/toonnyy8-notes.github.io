---
slug: flowgrpo-design-choices-and-results
title: Denoising Reduction、KL 與 Flow-GRPO 的實驗結果
cluster: RL, Reasoning & Self-Distillation
kind: topic
tags: [denoising-reduction, kl-regularization, reward-hacking, geneval, ocr, pickscore]
arxiv: ["2505.05470", "2403.03206", "2402.03300"]
sources: ["paper.txt Abstract, §4.2, §5, Appendix B, C"]
related: ["flowgrpo-lineage-hub", "flow-ode-to-sde-for-rl", "flowgrpo-denoising-mdp-and-grpo", "rl-finetuning-text-to-image-diffusion"]
---
# Denoising Reduction、KL 與 Flow-GRPO 的實驗結果

屬於 [Flow-GRPO 的研究脈絡總覽](flowgrpo-lineage-hub.html)。數字取自論文原文。

## Denoising Reduction

訓練時取樣只用 $T=10$ 步，推論仍用原本步數（SD3.5-M 為 $T=40$）。論文 §5.3：資料收集步數由 40 降到 10，三個任務都有超過 4 倍的訓練加速且最終獎勵不變；再降到 5 不一定更快，有時更慢。

## 任務與結果

以 SD3.5-M（2403.03206）為基礎模型，三類任務：GenEval 組合生成（可驗證獎勵）、視覺文字渲染（規則式）、以 PickScore 為獎勵的人類偏好對齊（模型式）。GenEval 由 63% 升到 95%（論文稱優於 GPT-4o），文字渲染由 59% 升到 92%。

## KL 與 reward hacking

論文以 KL 正則防止獎勵上升時品質或多樣性下降：GenEval 與 OCR 去掉 KL 會使 DrawBench 的品質與偏好分數明顯下降；PickScore 任務去掉 KL 則多樣性崩潰，輸出趨向單一風格。論文稱適當的 KL 可達到與無 KL 相近的獎勵，但訓練較久。

## 其他超參數與消融

- 附錄 B.3：G=24、a=0.7、解析度 512、LoRA（alpha=64、r=32）；$\beta$ 為 0.04（GenEval、文字渲染）與 0.01（PickScore）。
- 雜訊強度 $a$：0.1 探索不足；到 0.7 加速；0.7 到 1.0 無額外效益；過大使影像品質下降、獎勵為零（見 [ODE-to-SDE 轉換](flow-ode-to-sde-for-rl.html)）。
- 群組大小：G=12 與 6 訓練不穩並崩潰，G=24 穩定；論文稱與 ProRL、AceReason-Nemotron 的觀察一致。

GRPO 目標見 [Denoising 作為 MDP，以及 Flow-GRPO 的目標函數](flowgrpo-denoising-mdp-and-grpo.html)，比較方法見 [文字生成圖像擴散模型的 RL 微調脈絡](rl-finetuning-text-to-image-diffusion.html)。
