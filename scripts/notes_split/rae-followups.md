---
slug: rae-followups
title: RAE 後續改進與延伸：RAEv2、Scale-RAE、影片 RAE 與其他表徵編碼器路線
cluster: Diffusion & Generative Models
kind: topic
tags: [rae, raev2, scale-rae, text-to-image, video-tokenizer, self-guidance]
sources: [latent-diffusion-rae.md]
arxiv: ["2605.18324", "2601.16208", "2608.13556", "2607.14088", "2605.02134", "2608.05798", "2512.07829", "2603.19206", "2602.08620", "2510.18457"]
related: ["latent-diffusion-rae-overview", "representation-autoencoder-rae", "rae-distillation-drifting", "optimization-dilemma-semantic-alignment", "generative-eval-metrics"]
---
# RAE 後續改進與延伸：RAEv2、Scale-RAE、影片 RAE 與其他表徵編碼器路線

屬於 [Latent Diffusion 總覽](latent-diffusion-rae-overview.html)。原始方法見 [RAE](representation-autoencoder-rae.html)；蒸餾見 [RAE 蒸餾](rae-distillation-drifting.html)。

## RAEv2 / Improved Baselines（arXiv:2605.18324）
- **最後 k 層特徵求和**（Multi-Layer Sum, MLS）：x=Σ_{ℓ=L−K+1}^L z_ℓ，免 encoder fine-tune、免特殊資料即大幅改善 reconstruction。
- **RAE 與 REPA 非互斥而是互補**：大規模實證顯示同一表徵既可當 encoder 又可當中間層 distillation target（REPA 見 [語意/空間對齊](optimization-dilemma-semantic-alignment.html)）。
- **CFG / self-guidance**：原始 RAE 的 CFG 有問題、需另訓一個弱模型做 AutoGuidance；RAEv2 發現 REPA head h_φ 只用早層特徵做 x-prediction，本身就是「較弱模型」，把 full model 也 re-parameterize 成 x-prediction 即可用來做內部 guidance，省掉第二個模型。
- **效率**：80 epoch 即達 SOTA gFID 與 FDr⁶（相對舊 baseline 的 800 epoch），>10× 收斂加速，且在 T2I、world model 設定皆一致改善。RAEv2 已把 FDr⁶ 與 gFID 並列為主要報告指標（見 [生成評估指標](generative-eval-metrics.html)）。

## Scale-RAE / T2I（arXiv:2601.16208，rae-dit.github.io/scale-rae）
- SigLIP-2 So400M encoder；相對 FLUX VAE，Qwen2.5-1.5B + 2.4B DiT 從頭訓練達 **GenEval 4.0×、DPG-Bench 4.6× 收斂加速**（apples-to-apples，只差 latent space 與 decoder）。
- decoder 訓練資料：web+synthetic+text-rendering；**文字渲染場景需 targeted data composition**，純 scale 不足。 [arXiv](https://arxiv.org/abs/2601.16208v1)
- 縮放簡化：dimension-dependent noise schedule 仍關鍵，但 wide diffusion head 與 noise-augmented decoding 在大規模下效益可忽略。 [Quantum Zeitgeist](https://quantumzeitgeist.com/fidelity-scaling-text-image-diffusion-transformers/)
- VAE baseline 在高品質資料上約 64 epoch 後災難性過擬合，RAE 因高維語意 latent 有隱式正則化而穩定至 256+ epoch；DiT 從 0.5B 到 9.8B、LLM 從 1.5B 到 7B 皆一致優於 VAE。
- **對多模態架構的意涵**：共享語意 latent 讓 LLM 可直接處理生成的 latent（無需 decode 回 pixel）， [Rae-dit](https://rae-dit.github.io/scale-rae/) 並消除 VAE-based unified model 的「two-tower」設計； [The Moonlight](https://www.themoonlight.io/en/review/scaling-text-to-image-diffusion-transformers-with-representation-autoencoders) 作者展示 latent-space test-time scaling 可行。

## 影片對應
- V-RAE（arXiv:2608.13556）：凍結 VFM + lightweight temporal pooling 去時間冗餘 + video decoder； [arXiv](https://arxiv.org/html/2608.13556) 提出 tFVD 量測時間 latent 平滑度； [X](https://x.com/shumpeiMaxwell/status/2088416836023177371) 以四種凍結 encoder 評估，重申「好 reconstruction 不保證好 generation」。
- VideoRAE（arXiv:2607.14088）：用 V-JEPA2/VideoMAEv2 多尺度特徵 + 1D self-attention projector 壓成緊湊 latent。 [arXiv](https://arxiv.org/html/2607.14088v1)
- PV-VAE（arXiv:2605.02134）、KVAE tokenizer 家族（arXiv:2608.05798）為相關影片 tokenizer 路線。
- 限制：時間冗餘與壓縮更難。

## 其他表徵編碼器路線
- One Layer Is Enough（arXiv:2512.07829）：只需 adapt 預訓練 encoder 的一層即可用於生成。
- Both Semantics and Reconstruction Matter：讓 representation encoder 適配 T2I 與編輯。
- DINO-SAE（DINO spherical autoencoder）、VAE-REPA、RPiAE（arXiv:2603.19206，representation-pivoted AE，同時強化生成與編輯）、Improving Reconstruction of RAE（arXiv:2602.08620）、VFM-VAE（arXiv:2510.18457）等變體。
