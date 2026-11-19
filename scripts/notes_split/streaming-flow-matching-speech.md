---
slug: streaming-flow-matching-speech
title: 串流 flow matching 語音／音訊：CosyVoice 2、StreamFlow 與相關技術
cluster: Diffusion & Generative Models
kind: topic
tags: [flow-matching, streaming-tts, causal-attention-mask, cosyvoice2, streamflow, meanflow]
sources: [blockwise-smc-separation.md, causal-attention-diffusion.md]
arxiv: ["2412.10117", "2506.23986", "2505.22865", "2410.06885", "2505.13447", "2509.14858"]
related: ["causal-attention-diffusion-overview", "long-sequence-consistency-overview"]
---
# 串流 flow matching 語音／音訊：CosyVoice 2、StreamFlow 與相關技術

屬於 [Causal Attention × Diffusion 總覽](causal-attention-diffusion-overview.html) 與 [長序列一致性總覽](long-sequence-consistency-overview.html)。

## 真正融入因果性的串流案例
- **CosyVoice 2**（Du et al., Alibaba, arXiv 2412.10117）：chunk-aware **causal** flow matching decoder（擴散家族的 flow matching）。訓練時隨機抽樣四種注意力遮罩：non-causal（離線）、full-causal（僅過去、極低延遲）、chunk-M、chunk-2M（過去+少量未來）。這實現「單一模型同時支援串流與非串流」，並以多遮罩訓練達成隱式自蒸餾；前身用 look-ahead 卷積提供有限未來資訊；token overlap 機制維持跨 chunk 連續性。LibriSpeech test-clean WER 2.47%、speaker similarity 0.745，論文宣稱串流模式「virtually lossless synthesis quality in the streaming mode」（作者自報）。
- **StreamFlow**（Dake Guo, Jixun Yao, Linhan Ma, He Wang, Lei Xie，提交 2025-06-30，arXiv 2506.23986，NeurIPS 2025）：streaming flow matching + DiT。原文：「we propose StreamFlow, a novel neural architecture that facilitates streaming flow matching with diffusion transformers (DiT)... we introduce block-wise attention masks that enable the current block to receive information from the previous or subsequent block」。序列切塊，當前塊只能接收前/後鄰塊資訊；不同 DiT 層的遮罩階層式組合以調控感受野，採 OT-CFM（optimal-transport conditional flow matching）。並提出 Scale-DiT（adaptive layernorm + learnable residual scaling）。訓練用「causal noising」：固定過去 context、遮罩串流窗口以學習嚴格因果表徵。客觀指標（PESQ、STOI、ViSQOL、UTMOS）優於其他串流方法，接近非串流方法。
- **Causal Diffusion Models for Generalized Speech Enhancement / Stream.FM**（語音增強/復原）：frame-causal flow-based 模型，演算法延遲 32 ms、總延遲 48 ms，causal U-Net + causal convolution（僅在頻率軸降採樣以降延遲），無 look-ahead。
- **BinauralFlow**（arXiv 2505.22865）：causal U-Net + conditional flow matching 做串流雙耳語音合成，將雙耳渲染視為生成而非回歸問題。

## 其他 flow 端對應技術
- **F5-TTS（arXiv:2410.06885）/ E2-TTS**：非自回歸 flow matching + DiT，用 infilling（把參考音訊當條件延伸）。
- **MeanFlow（Geng, Deng, Bai, Kolter & He, arXiv:2505.13447）**：學 average velocity 而非 instantaneous velocity；MeanFlow-XL/2（676M 參數、240 epochs）「以單次函數評估（1-NFE）在 ImageNet 256×256 從頭訓練達 FID 3.43」（2-NFE 為 FID 2.93）；已延伸到語音增強（MeanFlowSE, arXiv:2509.14858）。對降低 posterior sampling 每步成本有潛力，但目前主要在無條件生成與語音增強，用於 inverse problem 的引導穩定性未經充分驗證。

以上多屬條件生成，尚未見用於 posterior sampling。

## 建議
- **音訊/語音方向**：採用 CosyVoice 2 的 chunk-aware causal flow matching 或 StreamFlow 的 block-wise causal attention mask 作為架構藍本——這是目前唯一在連續音訊上同時做到「擴散家族 + 顯式因果注意力遮罩 + 高品質」的成熟路線。注意 flow matching 是擴散的連續正規化流推廣，若審稿人嚴格區分「diffusion vs flow matching」需先界定。
- CosyVoice 2 的 WER/SS、StreamFlow 的相對優勢來自作者自報，缺乏獨立第三方複現。
