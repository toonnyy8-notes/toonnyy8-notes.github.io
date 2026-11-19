---
slug: llmmem-implicit-recurrent-state
title: 遞迴序列記憶：把歷史壓進固定大小的狀態
cluster: Architectures & Efficiency
kind: topic
tags: [recurrent, linear-attention, state-space-model, delta-rule, implicit-memory]
arxiv: ["2607.25380", "2006.16236", "2312.00752", "2405.21060", "2404.05892", "2307.08621", "2412.06464", "2605.22791", "2503.14456", "2510.26692", "2506.04761", "2603.15569", "2511.05963", "2502.13685", "2403.19887"]
sources: ["arXiv:2607.25380 §III-C, §III-D, Table I, Table II"]
related: ["delta-rule-deltanet-family", "gated-linear-recurrence-forgetting", "linear-attention-fast-weights", "llmmem-explicit-parameterized-test-time-memory", "llmmem-hybrid-management-evaluation", "llmmem-implicit-attention-access-control", "llmmem-taxonomy-overview", "llmmem-three-axes-taxonomy", "titans-lineage-overview"]
---
# 遞迴序列記憶

綜述（arXiv:2607.25380 §III-C）定義遞迴序列記憶為：把歷史壓縮成與前向計算耦合、持續演化的潛在狀態。實例包含 RNN、線性 attention 遞迴與狀態空間模型（SSM）。記憶是線上的、長期的（有效視野可涵蓋很長序列），但容量由狀態大小固定。

## 發展順序

論文依下列順序描述，每一項都是同一類狀態更新式的變體。

1. 經典 RNN 與 LSTM：以閘控資訊流延長有效視野，記憶完全經由狀態轉移存取。
2. 線性 attention（Katharopoulos 等，arXiv:2006.16236）：把 attention 核函數化並利用結合律，因果 attention 可寫成線性時間更新的遞迴狀態。更新式細節見 [Linear attention 與 fast weight programs](linear-attention-fast-weights.html)。
3. 資料相依的衰減與選擇性更新：Gated Linear Attention 在矩陣值狀態上加入衰減；RWKV-5/6（Eagle 與 Finch，arXiv:2404.05892）使用矩陣值狀態與動態遞迴；Mamba（arXiv:2312.00752）與 Mamba-2（arXiv:2405.21060）加入選擇性更新與 State Space Duality；RetNet（arXiv:2307.08621）提供遞迴與分塊兩種形式。遺忘機制見 [閘控線性遞迴](gated-linear-recurrence-forgetting.html)。
4. Delta rule：Gated Delta Networks（arXiv:2412.06464）以輸入條件的 delta rule 更新做到內容感知的保留，見 [Delta rule 與 DeltaNet](delta-rule-deltanet-family.html)。
5. 狀態編輯語義：Gated DeltaNet-2（arXiv:2605.22791）把 key 側抹除與 value 側寫入分開；Kaczmarz Linear Attention 把寫入視為 key 正規化的投影；Kalman Linear Attention 與 Gated KalmaNet 以濾波或線上迴歸依估計品質調節更新。RWKV-7（arXiv:2503.14456）以向量值狀態閘與 in-context learning rate 推廣 delta rule；Kimi Linear 的核心 KDA（arXiv:2510.26692）使用通道別對角衰減；DeltaProduct 與 Comba 以不同方式提高轉移的表達力。
6. 階層式容量：Log-Linear Attention（arXiv:2506.04761）用 Fenwick tree 階層讓狀態隨序列長度對數成長，近期資訊保留較細、遠期較粗，位於定長線性 attention 與逐 token KV 之間。
7. Mamba-3（arXiv:2603.15569）改進離散化，加入複數值更新與 MIMO 形式，仍是線上、定長儲存。
8. 混合與多狀態：Jamba（arXiv:2403.19887）交錯 attention 與 Mamba；MoM（arXiv:2502.13685）把 token 路由到多個遞迴狀態以降低過度壓縮與干擾。
9. 訓練誘發的信念狀態：Next-Latent Prediction（arXiv:2511.05963）加入預測未來潛在狀態的目標，論文歸為離線、由目標誘發的機制，沒有新增推論期寫入規則。

## 更新規則歸類

依 Table II，這一類大多屬 state-transition updates：記憶由學到的遞迴、選擇性掃描、delta rule 修正、投影或濾波式更新。各模型的差異在於每個 token 如何改變並分配壓縮後的狀態。

## 限制

論文指出長距離壓縮會降低保真度，且讀寫缺少獨立控制；Memory Caching、MoM 等方法主要改善單一序列內的保留，沒有提供跨 session 的持久記憶。這些限制是參數化外顯記憶的動機，見 [參數化外顯記憶與測試時更新](llmmem-explicit-parameterized-test-time-memory.html)。Titans 一系列論文如何從這條線走到測試時學習，見 [Titans 的研究脈絡總覽](titans-lineage-overview.html)；混合設計見 [混合架構、記憶管理與評估](llmmem-hybrid-management-evaluation.html)。

## 相關筆記

- [Attention 作為隱式記憶：KV cache、串流視窗與稀疏存取控制](llmmem-implicit-attention-access-control.html)
- [LLM 記憶機制的分類總覽：從 attention 與遞迴狀態到可定址的外顯記憶](llmmem-taxonomy-overview.html)
- [LLM 記憶的三條分類軸：representation、update dynamics、persistence](llmmem-three-axes-taxonomy.html)
