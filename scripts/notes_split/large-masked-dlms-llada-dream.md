---
slug: large-masked-dlms-llada-dream
title: 大型遮罩式 DLM：LLaDA、Dream 與相關系統
cluster: Diffusion Language Models
kind: topic
tags: [llada, dream, open-dcoder, parallel-decoding]
arxiv: ["2502.09992", "2508.15487", "2508.02193", "2506.17298", "2503.09573", "2412.15115", "2512.15745", "2512.15596", "2407.21783", "2512.09106"]
sources: ["paper:2512.15596 附錄 B 與第 2 節對該工作的描述", "arxiv.org/abs 頁面（標題、作者、年份）", "paper.txt §1, §4", "arXiv API metadata"]
related: [corrective-dlm-overview, mdlm-masked-diffusion-objective, code-benchmarks-for-revision, unmasking-policies-dllm-overview, confidence-thresholding-heuristics, unmasking-mdp-policy-design]
---
# 大型遮罩式 DLM：LLaDA、Dream 與相關系統

屬於 [CDLM 研究脈絡總覽](corrective-dlm-overview.html)與[unmasking policy 總覽](unmasking-policies-dllm-overview.html)。前置：[MDLM 訓練目標](mdlm-masked-diffusion-objective.html)。

## 論文引用的系統
| 系統 | 文獻 | arXiv | 在 CDLM 論文中的角色 |
| --- | --- | --- | --- |
| LLaDA | Nie et al.（2025），*Large Language Diffusion Models*，10 位作者 | 2502.09992 | 附錄描述為在相同設定下表現可與自回歸基線相當；LLaDA-8B-Base 是第 4 節的診斷對象，也是 8B 後訓練的起點 |
| Dream 7B | Ye et al.（2025），*Dream 7B: Diffusion Large Language Models*，8 位作者 | 2508.15487 | Dream-7B-Base 是第 4 節診斷對象；論文說明它由 Qwen2.5 自回歸檢查點繼續做遮罩擴散訓練而來 |
| Seed Diffusion | Song et al.（2025），22 位作者 | 2508.02193 | 論文列為 DLM 之一，並與 LLaDA 等同列為前綴補全型評測的對象 |
| Mercury | Inception Labs et al.（2025） | 2506.17298 | 論文列為 DLM 之一 |
| Block Diffusion | Arriola et al.（2025），8 位作者 | 2503.09573 | 出現在論文參考文獻中 |
| Qwen2.5 | Qwen（2024） | 2412.15115 | Dream-7B 與 Open-dCoder-0.5B 的自回歸起點（論文第 4 節） |
| LLaDA2.0 | Bie et al.（2025），31 位作者 | 2512.15745 | 論文參考文獻中的擴大規模工作 |

Open-dCoder-0.5B（Peng et al., 2025b）是 Open-dLLM 專案的模型，沒有 arXiv 論文，只有 GitHub 專案；它是 CDLM-0.5B 的預設起點，以 2000 步後訓練得到。

## 在 Learning Unmasking Policies…（arXiv 2512.09106）中的角色
論文第 1 節指出，LLaDA（Nie 等人，2025，2502.09992）與 Dream（Ye 等人，2025，2508.15487）在表現上可比擬同規模的自迴歸（AR）LLM。論文引用 Llama 3（Dubey 等人，2024，「The Llama 3 Herd of Models」，2407.21783）作為 AR 比較對象，並說明開源 dLLM 在推論效率上長期落後 AR 模型，直到 Fast-dLLM 才在 LLaDA 上達到比同規模 LLaMA 更高的 token 吞吐量（見 [Confidence 門檻式 heuristic](confidence-thresholding-heuristics.html)）。

在該論文中，這兩個模型是 MDP 的環境，參數凍結：實驗主要使用 LLaDA-8B-Instruct，Dream-7B-Instruct 用於附錄與遷移實驗。policy 約 30 萬參數，小於預訓練 dLLM 的 0.01%。

資料集：GSM8K（Cobbe 等人，2110.14168，「Training Verifiers to Solve Math Word Problems」）與 MATH 的訓練集合計約 15,000 筆用於訓練 policy，評估用 GSM8K 與 MATH-500。

## 對 CDLM 的意義
論文的觀察是：從頭預訓練（LLaDA）與由自回歸檢查點改訓（Dream、Open-dCoder）的 MDLM 都缺乏糾錯行為，因此問題來自監督型態，而不是特定的預訓練歷史。評測結果見 [程式碼評測基準](code-benchmarks-for-revision.html)。
