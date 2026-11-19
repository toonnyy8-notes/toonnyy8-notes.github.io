---
slug: weight-sharing-recursive-overview
title: 層間權重共享 × 層級 LoRA 鬆綁總覽
cluster: Architectures & Efficiency
kind: hub
tags: [weight-sharing, recursive-transformer, looped-transformer, lora, overview]
sources: [weight-sharing-lora.md]
arxiv: ["2502.17416", "2510.25741", "2604.11791", "2512.12880"]
related: ["relaxed-recursive-transformer", "looped-transformers-reasoning", "recursive-transformer-followups", "stages-of-inference-weight-sharing", "depth-nonuniform-lora-rank", "early-cross-layer-weight-sharing", "depth-redundancy-adjacent-methods", "lottery-ticket-redundancy-scaffold"]
---
# 層間權重共享 × 層級 LoRA 鬆綁總覽

## TL;DR
- **層間冗餘可以被壓縮掉，但目前壓縮的主要收益是「資源效率」（用更少參數與記憶體打平原有效能），而非「效能上界」的真正提升**——Bae et al. 的 [Relaxed Recursive Transformer](relaxed-recursive-transformer.html) 把 Gemma 2B 壓成約 1B 參數，並在 60B token uptraining + 知識蒸餾下逼近原模型（few-shot 平均 58.4% vs 58.6%），這是「以少打平」的最佳證據，但沒有超越原本 2B 模型的上界。
- **唯一有較強證據支持「能力提升」的維度是「推理密集任務」**：Saunshi et al.（arXiv:2502.17416, ICLR 2025）在受控 iso-FLOP 比較下顯示 looped 模型在推理任務上逼近甚至偶爾超越同運算量的非 looped 模型，但 perplexity 幾乎總是較差（見 [looped transformers](looped-transformers-reasoning.html)）；Ouro（arXiv:2510.25741）與 Geiping et al. 的 recurrent-depth 模型把「深度即運算」擴展到 test-time compute（見 [後續工作](recursive-transformer-followups.html)）。這些本質是「用同參數換更多迭代運算深度」，是把冗餘換成可迭代的運算，而非消滅冗餘後的免費午餐。
- **與 Lad et al. 四階段假說的交叉最關鍵的啟示是：U 形層敏感度（首尾脆弱、中間穩健）強烈暗示「共享程度與 LoRA rank 應隨深度非均勻分配」，但截至 2026 年 9 月，尚無論文直接以可解釋性為依據做深度非均勻的 rank 分配——這是明確且高價值的研究空白**（見 [四階段假說](stages-of-inference-weight-sharing.html) 與 [深度非均勻 rank](depth-nonuniform-lora-rank.html)）。

## 技術樹主幹
早期跨層共享（Universal Transformer 2018、ALBERT 2019、DEQ）→ 理論性 looped transformers（Giannou 2023、Fan 2024）→ 現代「轉換式」遞迴（Bae et al. RRT, 2024/ICLR 2025）→ 動態深度與 test-time compute（MoR 2025、Ouro 2025、Geiping 2025）。核心思想從「省參數」演進到「把深度當成一條獨立的 scaling 軸」。

## 優劣分析（可比較維度）

| 方法 | 參數量 | 每 token FLOPs | KV cache | 訓練成本 | 推論吞吐 | 硬體友善度 | 定位 |
|---|---|---|---|---|---|---|---|
| Vanilla | 基準 | 基準 | 基準 | 基準 | 基準 | 高 | — |
| 純層共享（ALBERT/CYCLE） | ↓↓ | 不變（深度不變） | 不變 | 需 uptraining | 需靠早退才提升 | 中（早退異質性問題） | 資源效率 |
| RRT（+LoRA rank r） | ↓（隨 r 回升） | 略增 | 不變 | +15–60B token | continuous depth-wise batching 理論 2–3× | 中 | 資源效率 |
| MoR | ↓（~1/3 唯一參數） | ↓（per-token 深度 + 稀疏 KV） | ↓（recursion-wise） | 同 FLOPs 更省 | 實測至 2.06× | 中高 | 資源效率（Pareto） |
| Looped（Saunshi/Ouro） | ↓↓ | ↑（多 loop） | 依實作 | 需從頭或大量 | loop 越多越慢 | 中 | 能力提升（推理） |
| Recurrent-depth（Geiping） | 中 | ↑↑（test-time 可調） | 小 context 友善 | 大 | 隨 loop 線性變慢 | 中 | 能力提升（test-time compute） |
| MoD | 不變 | ↓ | 不變 | 同基準 | 快 | 高（靜態圖） | 資源效率 |
| 深度剪枝 | ↓↓ | ↓↓ | ↓ | 極省（QLoRA heal，單卡） | 快 | 高 | 資源效率 |
| MoEUT | ↑（MoE 補參數） | 稀疏活化 | 不變 | 中 | 中 | 中 | 資源效率 |

**資源效率 vs 能力提升的證據強度**：
- **資源效率（證據強）**：RRT、MoR、深度剪枝、MoD、MoEUT 皆有明確 iso-FLOP/iso-param 對照顯示「以更少參數或更少 FLOPs 打平」。RRT 的 58.4% vs 58.6%、MoR 的 Pareto 前緣、Gromov 移除半層仍不掉分，皆屬此類。
- **能力提升（證據中等、且有邊界條件）**：僅 looped/recurrent 在「推理密集任務」有 iso-FLOP 下逼近或偶爾超越的證據（Saunshi et al. 的 34.3% vs 29.3%；Ouro 的知識操作提升）。但需注意：（i）perplexity 幾乎總是較差；（ii）「提升」多半來自「用同參數換更多序列運算/loop 深度」，屬 test-time compute 換取，而非消滅冗餘後的免費午餐；（iii）證據集中於中小模型與合成/數學任務。
- LoRA 鬆綁是把「純資源效率」往回退一步以換取更好的準確率（參數量回升，證據強）。

## 建議

**若目標是部署成本下降（資源效率）——證據充分，可直接採用：** 走轉換式路線（細節見 [RRT](relaxed-recursive-transformer.html)）；若 serving 吞吐是瓶頸，優先評估 MoR（實測 2.06×）而非 RRT 的理論早退（2–3× 為 oracle 模擬，未實測）。

**若目標是推理能力提升——證據有邊界，需先驗證：**
- 僅在推理/數學/程式碼為主的工作負載才考慮 looped/recurrent-depth；先在 iso-FLOP 下對照 vanilla，**門檻**：若 looped 在目標任務無法在 iso-FLOP 下追平非 looped（如 Saunshi 的開卷 QA / 數學字題模式），則不值得承受 perplexity 與延遲代價。
- 資源允許從頭訓大模型者可參考 Ouro（需 T 級 token 與穩定化）；資源有限者留在轉換式路線。

**研究者——最高價值的空白：** 見 [深度非均勻 rank × 遞迴 LoRA](depth-nonuniform-lora-rank.html)。

**會改變上述建議的觀測**：若未來出現「共享/遞迴模型在 iso-FLOP 下於一般（非推理）任務也穩定超越 vanilla」的強證據，則「能力提升」的判定應上修，遞迴路線的適用範圍應大幅擴張。

## 一般注意事項
- **perplexity vs 下游脫鉤**：looped 模型幾乎總是 perplexity 較差、下游推理較好，兩者不可互相推斷；比較時須指明用哪個指標。
- **規模外推風險**：多數遞迴/looped 的正面結果集中在 ≤3B 規模與合成/中小基準；是否外推到前沿規模仍未定（MoR 在 135M 甚至落後 vanilla）。
- **知識截止與新近性**：2026 年多篇（mechanistic 分析 arXiv:2604.11791、MoL arXiv:2512.12880、CoLa、LoopFormer、Ouro 等）為極新工作，部分尚未經完整同行評審，數字可能隨改版變動。
- 二手/彙整來源（Emergent Mind、Medium、Substack 等）僅用於交叉定位，數字皆回溯至一手論文。

## 建議閱讀順序
1. [早期跨層權重共享：Universal Transformer、ALBERT、DEQ](early-cross-layer-weight-sharing.html)
2. [Looped transformers 的理論與推理優勢](looped-transformers-reasoning.html)
3. [Relaxed Recursive Transformer（RRT）與層級 LoRA 鬆綁](relaxed-recursive-transformer.html)
4. [遞迴 Transformer 後續工作：MoR、Ouro、MoL、CoLa、recurrent-depth](recursive-transformer-followups.html)
5. [相鄰路線：MoD、深度剪枝、MoEUT](depth-redundancy-adjacent-methods.html)
6. [四階段推論假說與權重共享](stages-of-inference-weight-sharing.html)
7. [樂透票假說與「冗餘作為訓練鷹架」](lottery-ticket-redundancy-scaffold.html)
8. [深度非均勻 LoRA rank 分配：研究空白與方向](depth-nonuniform-lora-rank.html)
