---
slug: cql-conservative-policy-evaluation
title: CQL 的保守 policy evaluation：目標式 1、2 與下界定理 3.1、3.2
cluster: RL, Reasoning & Self-Distillation
kind: topic
tags: [CQL, conservative-Q, lower-bound, off-policy-evaluation]
arxiv: ["2006.04779"]
sources: ["arXiv:2006.04779 §3.1, Appendix C"]
related: ["rl-mdp-q-learning-basics", "offline-rl-distribution-shift", "cql-algorithm-variants-and-guarantees"]
---
# CQL 的保守 policy evaluation

CQL（arXiv:2006.04779）§3.1 先處理 off-policy evaluation：給定行為策略 `π_β` 產生的資料集 `D`，估計目標策略 `π` 的價值 `V^π(s)`。記號與 Bellman operator 見 [MDP、Q 函數與 Bellman backup](rl-mdp-q-learning-basics.html)，要解決的 OOD 動作高估問題見 [Offline RL 的分布偏移](offline-rl-distribution-shift.html)。

**式 1。** 在標準 Bellman error 外加一項 Q 值懲罰，`α` 為權衡係數：

`Q̂^{k+1} ← argmin_Q α E_{s~D, a~μ(a|s)}[Q(s,a)] + ½ E_{s,a~D}[(Q(s,a) − B̂^π Q̂^k(s,a))²]`

`μ(a|s)` 是被壓低 Q 值的動作分布。標準 Q 函數訓練不會在沒看過的狀態上查詢 Q，但會在沒看過的動作上查詢，所以論文把 `μ` 的狀態邊際限制為資料集的狀態分布：`μ(s,a) = d^{π_β}(s) μ(a|s)`。

**定理 3.1。** 對任何 `supp μ ⊂ supp π̂_β` 的 `μ`，以機率至少 `1−δ`，式 1 的不動點 `Q̂^π` 滿足 `Q̂^π ≤ Q^π − α(I−γP^π)^{-1}[μ/π̂_β] + (I−γP^π)^{-1} C_{r,T,δ} R_max / ((1−γ)√|D|)`。`α` 夠大時，對所有 `s∈D, a` 有 `Q̂^π ≤ Q^π`（逐點下界）；若 `B̂^π = B^π`，任何 `α>0` 皆成立。取樣誤差以 `|B̂^π − B^π| ≤ C_{r,T,δ}/√|D(s,a)|`（機率至少 `1−δ`）控制。

**式 2。** 若只需要 `E_π[Q̂^π]` 低估 `V^π`，可加上一項在資料分布 `π̂_β` 下最大化 Q 值，使下界更緊：

`α·E_{s~D,a~μ}[Q] − E_{s~D,a~π̂_β}[Q] + ½ E_{s,a,s'~D}[(Q − B̂^π Q̂^k)²]`

**定理 3.2。** 當 `μ = π` 時，`V̂^π(s) = E_π[Q̂^π(s,a)] ≤ V^π(s)`，條件為 `α` 大於論文給定的門檻（含 `Σ_a π(a|s)(π(a|s)/π̂_β(a|s) − 1)` 一項）；`B̂^π = B^π` 時任何 `α>0` 皆可。式 2 不保證逐點下界，因為它在 `π̂_β` 下抬高 Q 值，資料中常見的動作可能被高估。論文在 Appendix D.2 證明，最大化項換成 `π̂_β` 以外的分布時不保證下界。

論文另將結果推廣到線性與神經網路（以 NTK 為基礎）的函數近似（Theorem D.1、D.2），並指出資料量增加時所需的 `α` 下降。把式 2 組成完整演算法見 [CQL 的演算法變體與策略保證](cql-algorithm-variants-and-guarantees.html)。
