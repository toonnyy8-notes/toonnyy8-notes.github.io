---
slug: corrective-dlm-mixture-noise-posttraining
title: CDLM 的混合噪聲後訓練
cluster: Diffusion Language Models
kind: topic
tags: [cdlm, post-training, mixture-noise, error-aware-confidence]
arxiv: ["2512.15596"]
sources: ["paper:2512.15596 第 5、6 節"]
related: ["corrective-dlm-overview", "mdlm-masked-diffusion-objective", "dlm-remasking-self-correction", "uniform-noise-gidd-clean-supervision", "code-benchmarks-for-revision"]
---
# CDLM 的混合噪聲後訓練

屬於 [CDLM 研究脈絡總覽](corrective-dlm-overview.html)。本篇把前面各主題串起來。

## 定義
CDLM 是符合兩項條件的 DLM：(i) 對錯誤 token 的信心系統性低於正確 token；(ii) 用這個信心定位錯誤並迭代改善序列，做到就地修正。

## 目標函式（論文式 1）
先照 [MDLM](mdlm-masked-diffusion-objective.html) 遮罩一個隨機子集 $；剩下的可見位置 $ 各以機率 $\alpha$ 換成均勻抽取的錯誤 token，得到集合 $。

3744209L=\mathbb{E}\big[L_{mask}+\lambda_{noise}\tfrac{1}{|N|}\textstyle\sum_{i\in N}\ell_i\big],\quad \ell_i=-\log p_\theta(x^\star_i\mid z)3744209

乾淨可見 token（\setminus N$）沒有損失。論文所有語言模型實驗使用 $\alpha=0.1$、$\lambda_{noise}=0.1$。

## 實驗要點（0.5B，對照吸收式 MDLM）
- 設定：Open-dCoder-0.5B 於 Nemotron-SFT-Code 後訓練 2000 步；與 MDLM 僅目標不同。
- Gap 在 15 個格子中有 14 個大於 MDLM；Hit@K 在 K=1..6 都較高。
- Pass@1 在各 T 均高於 MDLM 與基底，與 MDLM 的差距由 T=1 的 +0.026 擴大到 T=2–4 的 +0.078 至 +0.085。
- 效果可轉移到 LLaDA-8B；8B 時兩種微調都沒有改善從頭生成。0.5B 在 MBPP(+) 仍低於基底。

與 [GIDD](uniform-noise-gidd-clean-supervision.html) 的差別、與 [PRISM 等修正方法](dlm-remasking-self-correction.html) 的關係，以及評測方式見對應筆記與 [CRB](code-benchmarks-for-revision.html)。
