---
slug: rzero-self-training-collapse
title: 自我訓練的崩壞風險：model collapse 與 reward hacking
cluster: RL, Reasoning & Self-Distillation
kind: topic
tags: [model-collapse, synthetic-data, self-training, diversity-collapse]
arxiv: ["2508.05004", "2404.05090", "2410.04840", "2311.16822", "2505.21444", "2509.15194"]
sources: ["arXiv 2508.05004 §4.2、附錄 E", "arXiv abstract 2404.05090", "arXiv abstract 2410.04840", "arXiv abstract 2311.16822", "arXiv abstract 2505.21444", "arXiv abstract 2509.15194"]
related: ["rzero-lineage-hub", "rzero-majority-vote-pseudo-labels", "rzero-challenger-solver-pipeline"]
---
# 自我訓練的崩壞風險

R-Zero §4.2 觀察到：各規模模型在前幾輪迭代提升，之後效能下降；模型越大，下降越晚。0.6B 在第一輪（Step 15）達到峰值，4B 持續到第三輪，Step 60 才明顯下降。論文認為這顯示框架本身有不穩定性，並未宣稱已解決。

附錄 E 分析偽標籤品質：偽標籤正確率隨迭代下降，但沒有統一的臨界值，0.6B 在正確率仍有 70.6% 時開始下降，4B 在 48.8% 才開始下降，因此雜訊比例不是唯一原因。論文推測另一個原因是只用自產資料訓練時出現的 model collapse，即多樣性喪失與自身偏誤放大，並引用下列文獻。以下依摘要記錄：

- Seddik et al.（arXiv 2404.05090）：以統計模型證明只用合成資料訓練無法避免 collapse；混合真實與合成資料時，給出可避免 collapse 的合成資料上限估計。
- Dohmatob et al.（arXiv 2410.04840，Strong Model Collapse）：即使合成資料只占訓練集約 1%，仍可能導致 collapse；簡化設定下，較大的模型可能放大 collapse。
- Briesch et al.（arXiv 2311.16822）：分析 LLM 自消耗訓練迴圈（標題：Large Language Models Suffer From Their Own Output）。
- 與 R-Zero 較直接相關的 RL 設定：Shafayat et al.（arXiv 2505.21444）報告長時間以多數決自我獎勵訓練會出現 reward hacking 與突然崩壞；EVOL-RL（arXiv 2509.15194）命名了 selection-induced diversity collapse。兩者見 [多數決偽標籤](rzero-majority-vote-pseudo-labels.html)。

Shumailov et al. 發表於 Nature 的 model collapse 論文與 Tan et al.（2024）等亦在 R-Zero 的引用清單中；本筆記未找到可核對的 arXiv 編號，故不收錄。R-Zero 對應的防護只有題目過濾與重複懲罰，見 [R-Zero 的 Challenger–Solver 迭代流程](rzero-challenger-solver-pipeline.html)。
