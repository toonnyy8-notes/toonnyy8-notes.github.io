---
slug: duet-posterior-sampling-separation
title: Duet 式分段 autoregressive posterior sampling 與 diffusion 音源分離方法
cluster: Diffusion & Generative Models
kind: topic
tags: [source-separation, posterior-sampling, duet-singing, msdm, arraydps, undiff, dgmo, identity-drift]
sources: [blockwise-smc-separation.md]
arxiv: ["2311.07345", "2302.02257", "2409.12346", "2505.23305", "2505.05657", "2506.02858", "2505.23625"]
related: ["long-sequence-consistency-overview", "blockwise-smc-time-axis"]
---
# Duet 式分段 autoregressive posterior sampling 與 diffusion 音源分離方法

屬於 [長序列一致性總覽](long-sequence-consistency-overview.html)。

## Duet Singing Voices Separation（Yu et al., 2023, arXiv:2311.07345）
與長序列 zero-shot 分離最相關的工作。
- 發現單一 unconditional 單歌手 DM 做 posterior sampling 時，分離出的 singer identity 會「每隔一小段就從一個歌手切換到另一個」（因為 prior 只提供隱式音色連貫與音高分布）。
- 解法：把混音切成 overlapping segments，**sequentially/autoregressive** 做 posterior sampling，將「當前段混音 + 前一段已分離結果的重疊部分」當條件，等於把前段結果當 inpainting condition 延伸。用 Mariani et al. 的 weakly-supervised posterior score。
- 在 MedleyVox 資料集（Jeon et al. 2023；來自 23 首 MedleyDB v1+v2 歌曲的 381 段 / 約 1.1 小時）以不同 overlap ratio 測試，「the proposed method outperforms naive posterior sampling baseline」。
- 是否重訓：**否**（用預訓練單源 DM）。能否 posterior sampling：**是**——這正是 inverse problem 設定。計算成本：段數 × 每段 posterior sampling；overlap 越大越貴但越連貫。
- 缺點：單軌跡貪婪傳遞，早期錯誤會累積（無回溯機制）；仍可能漂移。以 SMC 補足的提案見 [沿時間軸的 block-wise SMC](blockwise-smc-time-axis.html)。

## 其他 diffusion 分離 / inverse problem 方法
已用於 posterior sampling 的一致性技術，明確者僅 Duet。其他方法：
- **MSDM**（Mariani et al., ICLR 2024, arXiv:2302.02257）：學 4 源（Bass/Drums/Guitar/Piano）聯合分布，用 Dirac likelihood 分離；但在固定片段上，跨長序列一致性靠聯合 prior 隱式維持，未系統處理漂移。後繼有 MSG-LD（arXiv:2409.12346）、MGE-LDM（arXiv:2505.23305）等 latent 版本。
- **ArrayDPS**（Xu et al., ICML 2025, arXiv:2505.05657）：多通道 blind separation，用 DPS + IVA 初始化估計 relative RIR；array-agnostic、unsupervised，SDR 上超越所有 unsupervised baseline 並可比擬 supervised，但一致性來自空間 RTF 而非時間軸機制。
- **UnDiff**（Iashchenko et al., 2023）：首個通用音訊 inverse problem diffusion 框架（含分離、bandwidth extension、vocoding、declipping），但論文自陳分離有 failure case。
- **DGMO**（Lee et al., Interspeech 2025, arXiv:2506.02858）：training-free，用預訓練 audio diffusion 做 spectrogram mask 優化（test-time optimization）解 language-queried 分離，非 posterior sampling，且逐段處理。相關的 ZeroSep（arXiv:2505.23625）用 diffusion inversion 做 zero-shot 分離。

## 分段 posterior sampling 維持 source identity 的現有做法與優缺點

| 做法 | 優點 | 缺點 |
|---|---|---|
| overlapping segments + AR conditioning（Duet） | training-free、直接、可人在環 | 單軌跡貪婪、早期錯誤累積、無回溯 |
| speaker/instrument embedding 條件化 | 明確錨定 identity | 需要 embedding 模型，且 embedding 需在 zero-shot 下可靠 |
| joint prior（MSDM） | 跨源相關性強 | 需 joint 訓練、源數固定、不可任意泛化 |

## 評估指標爭議
generative separation 的 BSS-Eval（SDR/SIR/SAR）對相位與非線性失真敏感、對生成式系統可能不可靠；量化 identity drift 尚無標準指標，需自行設計（如 embedding 段間相似度）。
