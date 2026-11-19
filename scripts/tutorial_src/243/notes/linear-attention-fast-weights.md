---
slug: linear-attention-fast-weights
title: Linear attention 與 fast weight programs：矩陣值記憶的起點
cluster: Architectures & Efficiency
kind: topic
tags: [linear-attention, fast-weights, associative-memory]
arxiv: ["2501.00663", "2006.16236", "2102.11174", "1706.03762"]
sources: ["arXiv:2501.00663 §1, §2.1, Appendix A.3", "arXiv:2006.16236 abstract", "arXiv:2102.11174 abstract"]
related: ["delta-rule-deltanet-family", "gated-linear-recurrence-forgetting", "ttt-linear-mlp"]
---
# Linear attention 與 fast weight programs

**Linear attention**（Katharopoulos 等，Transformers are RNNs，arXiv:2006.16236）把 softmax 換成核函數 `φ`，使 `φ(x,y)=φ(x)φ(y)`。Titans 的式 3 把輸出改寫為 `φ(Q_i)^T Σ φ(K_j)V_j / φ(Q_i)^T Σ φ(K_l)`，其中的和可以逐步累加。核取恆等時成為遞迴 `M_t = M_{t-1} + K_t^T V_t`、`y_t = Q_t M_t`（式 4、5），推論成本與長度成線性。

Titans 指出其代價：加法寫入使長序列的記憶溢出，且固定大小的矩陣無法容納很長的歷史。

**Fast weight programs** 是這種矩陣記憶的更早來源。Titans 的敘述是：把線性層視為 key-value 聯想記憶，可追溯到 Schmidhuber 1992 的 fast weight programs，其中動態的快速權重充當可寫入記憶；常用的學習規則有 Hebbian 與 delta。Schlag、Irie、Schmidhuber 的 Linear Transformers Are Secretly Fast Weight Programmers（arXiv:2102.11174）建立了 linear attention 與 fast weight 的對應。Schmidhuber 1992 與 Hebb、Widrow-Hoff 的原始文獻不在 arXiv，此處僅轉述 Titans 的描述。

Titans 對這條線的批評：這些模型以當下的 surprise 更新、多數沒有遺忘閘，且記憶為淺層矩陣。後續改進見 [閘控遞迴](gated-linear-recurrence-forgetting.html) 與 [Delta rule](delta-rule-deltanet-family.html)；以梯度更新整個子網路的路線見 [原版 TTT](ttt-linear-mlp.html)。
