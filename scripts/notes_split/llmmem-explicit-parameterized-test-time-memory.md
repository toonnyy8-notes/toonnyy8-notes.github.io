---
slug: llmmem-explicit-parameterized-test-time-memory
title: 參數化外顯記憶：fast weights、Titans、TTT-E2E 與測試時更新
cluster: Architectures & Efficiency
kind: topic
tags: [fast-weights, test-time-training, titans, explicit-memory, parameterized-memory]
arxiv: ["2607.25380", "1610.06258", "2501.00663", "2512.23675", "2604.06169", "2402.04624", "2502.06049", "2601.12906", "2602.21204"]
sources: ["arXiv:2607.25380 §IV-A, §IV-D, §V-A, Table II"]
related: ["in-place-ttt", "llmmem-hybrid-management-evaluation", "llmmem-implicit-recurrent-state", "llmmem-multi-timescale-nested-updates", "llmmem-taxonomy-overview", "llmmem-three-axes-taxonomy", "recurrent-memory-segment-transformers", "titans-architectures-mac-mag-mal", "titans-lineage-overview", "titans-neural-memory-surprise-update", "ttt-e2e", "ttt-linear-mlp", "ttt-sequence-modeling-overview"]
---
# 參數化外顯記憶與測試時更新

綜述（arXiv:2607.25380 §IV-A）把這一類定義為：有專用的記憶參數，結構上與基礎模型權重分開，並由特定更新動態管理。記憶內容跨推論實例保留，可以只更新這一小部分參數。論文的判準是持久性與自主性，不是使用頻率；這類模組可以每個 token 都參與前向計算。

## 脈絡

1. fast weights：論文把「以短時間尺度快速更新的權重存放近期資訊、慢權重存放長期知識」的想法追溯到早期 fast-weight 公式，並引用 Ba 等人的 Using Fast Weights to Attend to the Recent Past（arXiv:1610.06258）。Memory-Augmented Neural Networks 與 Differentiable Neural Computers 進一步加入有可學習讀寫的顯式記憶矩陣。論文說明它們原本在較小規模提出。
2. 測試時參數更新：Titans（arXiv:2501.00663）引入專用記憶模組，推論時以 surprise 驅動的梯度訊號更新，只改記憶參數、不改主幹。機制細節見 [Titans 的神經長期記憶](titans-neural-memory-surprise-update.html)，與注意力的三種組合見 [Titans 的三種架構](titans-architectures-mac-mag-mal.html)，整條脈絡見 [Titans 的研究脈絡總覽](titans-lineage-overview.html)。TTT-E2E（arXiv:2512.23675）以可微的測試時目標與主幹聯合訓練，見 [TTT-E2E](ttt-e2e.html)；原版 TTT 層見 [TTT-Linear / TTT-MLP](ttt-linear-mlp.html)；整個 TTT 系列見 [Test-Time Training 總覽](ttt-sequence-modeling-overview.html)。
3. 不新增模組的 fast weight：In-Place TTT（arXiv:2604.06169）把 FFN 的下投影矩陣當可重用的 fast-weight 記憶，按輸入分塊更新，見 [In-Place TTT](in-place-ttt.html)。論文的重點是顯式記憶不一定是視覺上獨立的記憶庫，原本靜態的參數矩陣被賦予自主寫入語義時也成立。
4. 寫入控制：Gated Differentiable Working Memory（GDWM，arXiv:2601.12906）依情境效益對測試時更新加閘，使寫入集中於可能改善長 context 預測的區塊。論文認為線上顯式記憶的難點不只是如何儲存，還有哪些觀察值得寫入持久或半持久狀態。
5. 槽位式變體：MEMORYLLM（arXiv:2402.04624）在 Transformer 各層嵌入固定大小的潛在記憶池，讀到新知識時以只有前向的自我更新重寫記憶 token；LM2（arXiv:2502.06049）在 Transformer 層加入以類 cross-attention 存取並由閘控調節的專用記憶槽。兩者都表示顯式線上記憶不一定靠梯度優化。
6. 解耦學習動態：主幹編碼一般語言能力，記憶參數承載情境或任務資訊，更新記憶不覆寫預訓練表示；代價是容量上限、更新穩定性與對暫態訊號過擬合的風險。

## 與隱式記憶的邊界

論文引用分析指出，一大類以 key–value 綁定做內層更新的 TTT 可以改寫成學到的線性 attention（arXiv:2602.21204）。這不否定記憶詮釋，但意味著只有具備明確指定、自主更新的儲存基底的變體，才在主分類中算顯式記憶。

## 更新規則

Titans、TTT-E2E、In-Place TTT 都是 optimization-based writing；GDWM 在其上加上 signal-gated 控制；MEMORYLLM 與 LM2 屬槽位或前向更新。對應的風險是漂移、不穩定與目標不一致。多時間尺度的組織方式見 [多時間尺度與巢狀更新](llmmem-multi-timescale-nested-updates.html)；與分段記憶 Transformer 的關係見 [分段 Transformer 與外掛記憶](recurrent-memory-segment-transformers.html)。組合進完整架構的方式見 [混合架構、記憶管理與評估](llmmem-hybrid-management-evaluation.html)，其前身是 [遞迴序列記憶](llmmem-implicit-recurrent-state.html)。

## 相關筆記

- [LLM 記憶機制的分類總覽：從 attention 與遞迴狀態到可定址的外顯記憶](llmmem-taxonomy-overview.html)
- [LLM 記憶的三條分類軸：representation、update dynamics、persistence](llmmem-three-axes-taxonomy.html)
