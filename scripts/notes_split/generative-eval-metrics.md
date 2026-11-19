---
slug: generative-eval-metrics
title: 影像生成評估指標的演進：gFID 之外（FDr⁶、iFID、DiffusionBench）
cluster: Diffusion & Generative Models
kind: topic
tags: [evaluation, fid, fdr6, ifid, diffusionbench]
sources: [latent-diffusion-rae.md]
arxiv: ["2604.28190", "2603.05630", "2606.24888"]
related: ["latent-diffusion-rae-overview", "diffusability-latent-regularizers", "pixel-space-diffusion", "cdlm-benchmarks-and-eval"]
---
# 影像生成評估指標的演進：gFID 之外（FDr⁶、iFID、DiffusionBench）

屬於 [Latent Diffusion 總覽](latent-diffusion-rae-overview.html)。

- **gFID 的侷限**：只用 Inception 特徵空間，易被 Goodhart 化（部分 SOTA 模型 FID 已低於真實 validation set 卻仍可辨為假）。 [WisPaper](https://www.wispaper.ai/en/user-blog/frechet-loss-visual-generation-20260506/eng)
- **FDr⁶**（Representation Fréchet Loss，arXiv:2604.28190）：對六個表徵空間（Inception-v3、ConvNeXt-v2、DINOv2、MAE、SigLIP2、CLIP）各自以 validation FD 正規化後取算術平均；並發現 FD 可作為訓練目標（decouple 50k 估計 population 與 1024 batch）， [arXiv](https://arxiv.org/abs/2604.28190) 甚至可把多步 diffusion 免蒸餾轉一步。FDr-CLIP 另作 held-out 評估器。
- **FD_DINOv2 / FDr**：RAEv2 已把 FDr⁶ 與 gFID 並列為主要報告指標。
- **iFID**（arXiv:2603.05630）：以最近鄰插值衡量流形連續性，跨 VAE Pearson/Spearman ≈0.85（見 [其他 diffusability 觀點](diffusability-latent-regularizers.html)）。
- **DiffusionBench**（arXiv:2606.24888）：NanoGen 統一框架同時支援 RAE/VAE/pixel-space/MeanFlow，橫跨 ImageNet 與 T2I。 [arXiv](https://arxiv.org/abs/2606.24888) 關鍵反面發現：21 個 latent diffusion 方法在 ImageNet 與 T2I 上的排名**無正相關（Pearson −0.377 至 −0.580）**，代表只在 ImageNet 上優化可能無法遷移到 T2I。 [Hugging Face](https://huggingface.co/papers/2606.24888) pixel-space 方面的發現見 [pixel-space 第三路線](pixel-space-diffusion.html)。

## 建議
報告時以 **FDr⁶ + iFID** 補充 gFID；若做 T2I，DiffusionBench 是目前唯一橫跨兩任務的統一協定。

## 未解問題
FDr⁶ 仍繼承 Fréchet 的 Gaussian 假設；當多個 SOTA 已「FID 低於真實 validation」，需要超越 moment-matching 的分布距離或與人評對齊的自動指標。

語言模型端的類似問題（Gen PPL 被操縱）見 [連續擴散 LM 的評估陷阱](cdlm-benchmarks-and-eval.html)。
