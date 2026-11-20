---
slug: rswp-distribution-sharpening-debate
title: 分布銳化之爭：RL 後訓練是否只讓 base model 變尖
cluster: RL, Reasoning & Self-Distillation
kind: topic
tags: [distribution-sharpening, pass@k, RLVR, diversity]
arxiv: ["2510.14901", "2506.02355", "2506.10947", "2504.13837", "2509.06941"]
sources: ["arXiv:2510.14901 §1, §5.3", "arXiv abstract 2506.02355", "arXiv abstract 2506.10947", "arXiv abstract 2504.13837", "arXiv abstract 2509.06941"]
related: ["rswp-rlvr-grpo-background", "rswp-power-vs-low-temperature", "rswp-power-sampling-algorithm-results"]
---
# 分布銳化之爭：RL 後訓練是否只讓 base model 變尖

2510.14901 §1 以一個問題起頭：RL 後訓練出現的能力，是否為 base model 沒有的新行為。論文稱此為 distribution sharpening 問題：後訓練分布是否只是 base model 分布的「較尖」版本，而不是把機率放到 base model 不太會生成的推理軌跡上。

論文引用的四項工作：

- He、Fried、Welleck（2506.02355）：論文引用它與 Song 等人比較 base model 與後訓練模型的 pass@k，指出 k 大時 base model 反而較佳，後訓練模型生成多樣性下降。
- Shao 等人（2506.10947，Spurious Rewards）：與 He 等人一同被論文列在 sharpening 問題的文獻中；本文在 §2 另提到該文使用隨機獎勵。
- Yue 等人（2504.13837）：論文指出該文觀察到 RL 後的推理軌跡集中在 base model 下的高概似度與高信心區域；本文以 Figure 4 在自己的實驗重現此現象。
- Song、Kempe、Munos（2509.06941，Outcome-based Exploration）：與 He 等人一同被引用於 pass@k 與多樣性下降的結論。

論文由此推論：RL 把 pass@k 的表現重新分配到單次作答表現，代價是多次作答的表現。論文的設計因此是顯式指定一個「較尖」的目標分布（見 [冪次分布與低溫取樣](rswp-power-vs-low-temperature.html)），若 RL 後模型只是銳化後的 base model，取樣該分布應得到類似效果。結果見 [冪次取樣演算法與結果](rswp-power-sampling-algorithm-results.html)：GRPO 的樣本集中在概似度最高的峰值，冪次取樣的 pass@k 在 k > 1 時優於 GRPO，高 k 時與 base model 相當。

背景：[RLVR 與 GRPO](rswp-rlvr-grpo-background.html)。
