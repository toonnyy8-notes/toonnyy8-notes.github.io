---
slug: uniform-noise-gidd-clean-supervision
title: 均勻噪聲、GIDD 與是否監督乾淨 token
cluster: Diffusion Language Models
kind: topic
tags: [gidd, uniform-noise, bert, supervision-target]
arxiv: ["2503.04482", "2412.10193", "2510.03280", "1810.04805", "2512.15596"]
sources: ["paper:2512.15596 附錄 B 與第 2 節對該工作的描述", "arxiv.org/abs 頁面（標題、作者、年份）"]
related: ["corrective-dlm-overview"]
---
# 均勻噪聲、GIDD 與是否監督乾淨 token

屬於 [CDLM 研究脈絡總覽](corrective-dlm-overview.html)。

## 均勻噪聲的位置
- Schiff et al.（2025；arXiv 2412.10193），*Simple Guidance Mechanisms for Discrete Diffusion Models*，10 位作者。論文描述：均勻噪聲離散擴散較易引導，因為每個 token 都能被修改。
- Ni et al.（2025；arXiv 2510.03280），*Training Optimal Large Diffusion Language Models*，8 位作者。論文引用其結論：均勻噪聲在語言建模品質上仍落後吸收狀態擴散；並引用其觀察，均勻 DLM 不需被告知哪些位置受損就能學會分辨噪聲與乾淨 token，CDLM 把這種可分辨性帶入 MDLM。

## GIDD
von Rütte et al.（2025；arXiv 2503.04482），*Generalized Interpolating Discrete Diffusion*，6 位作者。論文說明它同樣結合吸收與均勻噪聲，所以「兩種噪聲並用」不是 CDLM 的貢獻。論文附錄 Table 6 列出差異：

| 面向 | GIDD | CDLM |
| --- | --- | --- |
| 框架 | 廣義內插擴散 | 標準 MDLM |
| 訓練 | 原論文從頭訓練 | 對既有 MDLM 後訓練 |
| 損失 | 統一 ELBO（KL + Itakura–Saito） | MDLM 損失加可分離噪聲項 $\lambda_{noise}$ |
| 乾淨 token 損失 | 有（監督所有位置） | 無（只監督遮罩與受損位置） |
| 解碼 | 每步任何 token 都可能改變 | remask-then-repredict，局部就地修正 |

## BERT 式訓練
Devlin et al.（2019；arXiv 1810.04805），*BERT*，4 位作者。BERT 的遮罩語言模型與論文所稱的 BERT 式訓練同屬「監督所有位置」一類。CDLM 第 6.2 節加入乾淨 token 項（權重 ctw = 0.1）後，refinement 從小幅增益變成下降：Pass@1 由 T=1 的 0.204 降到 T=4 的 0.182，而 CDLM 由 0.197 升到 0.205。
