---
slug: revflow-few-step-native-matching
title: 少步生成與表示空間的分佈匹配：RevFlow 的原生匹配
cluster: Diffusion & Generative Models
kind: topic
tags: [few-step, meanflow, consistency-model, distribution-matching, fd-loss]
arxiv: ["2611.34782", "2202.00512", "2505.13447", "2604.28190", "2607.02375", "2303.01469"]
sources: [arxiv-abstracts, paper-2611.34782-related-work]
related: ["fmlm-flow-map-distillation-lineage", "fewer-steps-sampling-acceleration", "revflow-rae-decoder-burden", "revflow-lineage-overview"]
---
# 少步生成與原生分佈匹配

論文把少步生成分為三條來源：蒸餾（Progressive Distillation 2202.00512、分佈匹配蒸餾）、consistency（Consistency Models 2303.01469）、平均速度建模（MeanFlow，2505.13447）。另一條是在固定特徵空間做分佈匹配，從 MMD 到近期以預訓練表示定義的目標：Representation Fréchet Loss（Yang 等，2604.28190）與 Representation Distribution Matching（Feng 等，2607.02375）。

論文指出：latent 空間的生成器要做表示匹配，必須先把生成的 latent 解碼成像素，再用編碼器重新編碼；RevFlow 直接生成編碼器表示，可在表示空間原生匹配（native matching, NM），不需解碼再編碼。論文報告 NM 同時改善單階段 RAE 與兩階段 RevFlow；短暫的 NM 調適後，RevFlow-XL 以每階段 1 步（共 2 NFE）、不用 CFG 得到 2.85 FID。

MeanFlow 等少步方法的發展整理見 [Flow map 與少步蒸餾的發展](fmlm-flow-map-distillation-lineage.html) 與 [減少取樣步數與加速訓練的相關工作](fewer-steps-sampling-acceleration.html)。
