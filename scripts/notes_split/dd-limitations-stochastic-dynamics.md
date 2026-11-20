---
slug: dd-limitations-stochastic-dynamics
title: Decision Diffuser 的限制與隨機動力學
cluster: RL, Reasoning & Self-Distillation
kind: topic
tags: [limitations, stochastic-dynamics, return-conditioning, pomdp]
arxiv: ["2211.15657", "2210.13435", "1912.13465", "1912.02875"]
sources: ["paper.txt §5, Appendix G, M", "2210.13435 僅依本論文描述與 arXiv API 題名", "arXiv API metadata"]
related: ["decision-diffuser-lineage-hub", "return-conditioned-sequence-modeling", "composing-constraints-and-skills", "dd-experiments-and-ablations", "inverse-dynamics-state-diffusion"]
---
# Decision Diffuser 的限制與隨機動力學

屬於[總覽](decision-diffuser-lineage-hub.html)。

## 報酬條件的一般問題
報酬條件策略的前身包括 Reward-Conditioned Policies（Kumar 等，arXiv 1912.13465）與 Reinforcement Learning Upside Down（Schmidhuber，arXiv 1912.02875），整理見[報酬條件的監督式學習](return-conditioned-sequence-modeling.html)。附錄 G 指出此類策略在環境隨機性增加時變差，因為無法區分好動作帶來的高回報與環境隨機性帶來的高回報。

## 附錄 G 的實驗
Block Push（力矩控制），以機率 p 以隨機動作取代策略動作。成功率（p=0、0.05、0.10、0.15）：
- DD：84.7、77.3、53.2、41.3
- Diffuser：72.4、63.2、52.3、41.6
- CQL：73.2、61.8、51.2、42.2
- BC：55.2、49.3、25.8、15.1

p=0.05 時 DD 優於 Diffuser 與 CQL；p 較大時各方法趨近。論文提到 Yang 等（2022，arXiv 2210.13435，"Dichotomy of Control"）與 Villaflor 等（2022）以潛在的未來狀態而非回報作為條件，並將此列為未來方向。

## 附錄 M 的限制
- 假設完全可觀察的 MDP，延伸到 POMDP 可能產生自我欺騙（Ortega 等 2021）。
- 僅離線，不能探索與線上更新；論文建議參考 Zheng 等（2022）。
- 只在狀態型環境實驗；影像環境可考慮潛在空間擴散（Rombach 等 2022）。
- 只支援 AND 與 NOT 組合（見[組合限制與技能](composing-constraints-and-skills.html)）。
- 隨機動力學下優勢消失；資料量有限時，論文提到擴散模型容易過擬合。
