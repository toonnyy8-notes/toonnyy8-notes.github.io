---
slug: aniso-spectral-weight-analysis
title: 學到的權重分析：誤差放大、PCA 基底與 w ∝ 資料方差
cluster: Diffusion & Generative Models
kind: topic
tags: [spectral-bias, pca, error-amplification, conditioning, anisoflow]
arxiv: ["2611.17767", "2503.03206", "2603.02337", "2603.06454", "2602.02908", "2605.27102"]
sources: [2611.17767 section 5, 2503.03206 abstract, 2603.02337 abstract, 2603.06454 abstract, 2602.02908 abstract, 2605.27102 abstract]
related: ["aniso-lineage-overview", "aniso-target-normal-form-and-recovery", "aniso-quantization-conditioning", "diffusion-dynamics-theory-background", "loss-weighting-and-elbo"]
---
# 學到的權重分析

回到[總覽](aniso-lineage-overview.html)。

## 本論文的分析（Sec. 5）
- 還原映射可逆，所以對任何 w_i，最佳預測都還原成同一個條件平均速度 E[u | z_t, t]；w_i 不改變最小可達損失，只改變網路需要逼近的函數。
- 方向 i 在時間步 t 的還原後損失為 E[(û_W,i − u_W,i)² | t] / (α_t w_i + σ_t)²（Eq. 6）。w_i 增大使誤差被放大得較少，但目標含較多噪聲。
- 在「局部固定誤差」模型下，偏好權重 w_i*(t) = α_t C_i(t) / (σ_t A_i(t))（Eq. 7），A_i、C_i 為網路對噪聲與資料估計的誤差。若 A_i(t) = α_t q_i(t)Ā_i、C_i(t) = σ_t q_i(t)C̄_i，各時間步偏好同一權重 w_i* = C̄_i/Ā_i。
- 再假設噪聲估計誤差各方向相近、資料估計誤差正比於資料能量 λ_i，得 w_i ∝ λ_i，與 pixel 空間學到的 w_i ≈ cλ_i 一致（JiT-H/16 上 c = 0.15）。

## 相關的先前研究（本論文 Related Work 所列）
以下僅依各自摘要：
- **Wang & Pehlevan**（arXiv:2503.03206）：線性與卷積去噪器的學習動態有「逆方差譜律」，高方差方向比低方差方向早學到。
- **Preconditioned Flow Matching**（Ahamed 等，arXiv:2603.02337）：目標分佈共變異數病態時，梯度訓練快速擬合高方差方向、低方差方向進展慢；提出轉到較各向同性的表示再匹配。
- **Gagneux 等**（arXiv:2603.06454）：以數值研究比較損失加權與輸出參數化（噪聲、乾淨影像、速度）與資料流形維度的交互作用。
- **Wang 等**（arXiv:2602.02908）：以隨機矩陣理論分析線性設定下有限資料對學到的去噪器的影響。
- **Fu 等 JLT**（arXiv:2605.27102）：latent transformer 中，乾淨資料預測與速度預測的最佳解等價，但有限容量的網路在兩種介面下分配的計算不同。

加權視角的背景見 [Loss weighting 與 ELBO](loss-weighting-and-elbo.html) 與[擴散學習動態的理論背景](diffusion-dynamics-theory-background.html)。
