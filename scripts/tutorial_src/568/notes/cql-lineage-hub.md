---
slug: cql-lineage-hub
title: Conservative Q-Learning 的研究脈絡總覽
cluster: RL, Reasoning & Self-Distillation
kind: hub
tags: [CQL, offline-RL, lineage]
arxiv: ["2006.04779", "1312.5602", "1801.01290", "1802.09477", "1812.02900", "1906.00949", "1911.11361", "2005.01643", "2004.07219"]
sources: ["arXiv:2006.04779 §1, §2, §5, §6", "arXiv abstracts of cited works"]
related: ["rl-mdp-q-learning-basics", "offline-rl-distribution-shift", "cql-conservative-policy-evaluation", "cql-algorithm-variants-and-guarantees", "cql-vs-policy-constraint-methods", "cql-experiments-d4rl-atari"]
---
# CQL 研究脈絡總覽

Conservative Q-Learning（arXiv:2006.04779，Kumar、Zhou、Tucker、Levine，2020）針對 offline RL 中 value-based 方法的過度樂觀估計，提出學習保守的 Q 函數。閱讀順序如下。

1. [MDP、Q 函數與 Bellman backup](rl-mdp-q-learning-basics.html)：論文使用的記號，以及 DQN（arXiv:1312.5602）、SAC（arXiv:1801.01290）、TD3（arXiv:1802.09477）等 online 基礎。
2. [Offline RL 的分布偏移](offline-rl-distribution-shift.html)：為何以 OOD 動作做 Bellman backup 會高估 Q 值，以及 BCQ（arXiv:1812.02900）、BEAR（arXiv:1906.00949）、BRAC（arXiv:1911.11361）的策略限制作法；綜述為 arXiv:2005.01643。
3. [保守 policy evaluation](cql-conservative-policy-evaluation.html)：在 Bellman error 上加 Q 值懲罰，及其下界定理。
4. [min-max 形式、CQL(H)、CQL(ρ) 與策略保證](cql-algorithm-variants-and-guarantees.html)：完整演算法與 gap-expanding、safe policy improvement 定理。
5. [與策略限制等方法的比較](cql-vs-policy-constraint-methods.html)：CQL 不需估計行為策略的論點。
6. [實驗](cql-experiments-d4rl-atari.html)：D4RL（arXiv:2004.07219）與離線 Atari 的結果。

脈絡的年代順序見同資料夾的 timeline.json。
