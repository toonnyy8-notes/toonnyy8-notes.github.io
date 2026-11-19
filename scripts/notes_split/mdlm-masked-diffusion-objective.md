---
slug: mdlm-masked-diffusion-objective
title: MDLM：遮罩式擴散的訓練目標
cluster: Diffusion Language Models
kind: topic
tags: [masked-diffusion, elbo, mlm]
arxiv: ["2406.07524", "2512.15596"]
sources: ["paper:2512.15596 附錄 B 與第 2 節對該工作的描述", "arxiv.org/abs 頁面（標題、作者、年份）"]
related: ["corrective-dlm-overview", "discrete-diffusion-foundations-d3pm", "large-masked-dlms-llada-dream", "corrective-dlm-mixture-noise-posttraining"]
---
# MDLM：遮罩式擴散的訓練目標

屬於 [CDLM 研究脈絡總覽](corrective-dlm-overview.html)。前置：[離散擴散的基礎](discrete-diffusion-foundations-d3pm.html)。

## 來源
Sahoo et al.（2024；arXiv 2406.07524），*Simple and Effective Masked Diffusion Language Models*，作者共 8 人。CDLM 論文附錄 B.1 描述它把遮罩式擴散詮釋為「由加權的 MLM 式重建項組成的變分目標」，並在離散 DLM 中表現強。

## CDLM 論文使用的形式（第 2.1 節）
- 乾淨序列 =(x_1,\dots,x_n)$，吸收遮罩 token  \notin V$。
- 每個樣本抽 $\lambda\sim\mathrm{Uniform}[0,1]$，每個 token 以機率 $\lambda$ 獨立換成 $，得到 $。
- 損失 {absorb}(\theta)=\mathbb{E}_{x,z}\big[-\sum_{i:z_i=m}\log p_\theta(x_i\mid z)\big]$：只有被遮罩的位置有損失，可見位置的 logits 被忽略。

## 與 CDLM 的關係
論文的核心診斷來自這個目標的監督範圍：可見 token 沒有學習訊號，所以模型的信心不一定反映該 token 是否正確。CDLM 保留此項，再加上對受損可見 token 的項，見 [CDLM 的混合噪聲後訓練](corrective-dlm-mixture-noise-posttraining.html)。

論文指出 MDLM 這類模型是一整類 DLM 的基礎，列舉 LLaDA、Dream、Mercury、Seed Diffusion 等，見 [大型遮罩式 DLM](large-masked-dlms-llada-dream.html)。
