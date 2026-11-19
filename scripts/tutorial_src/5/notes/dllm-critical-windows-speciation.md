---
slug: dllm-critical-windows-speciation
title: critical window 與 speciation transition：生成結果在窄範圍的 noise level 定形
cluster: Diffusion & Generative Models
kind: topic
tags: [critical-window, speciation, phase-transition, diffusion-theory]
arxiv: ["2609.20539", "2312.06205", "2305.19693", "2402.18491", "2402.16991", "2403.01633"]
sources: [2609.20539 related work, arXiv titles]
related: ["dllm-parallelism-lineage-hub", "dllm-score-oracle-lower-bounds", "dllm-parallelism-separation-result"]
---
# critical window 與 speciation transition

屬於 [dLLM 平行生成的理論脈絡總覽](dllm-parallelism-lineage-hub.html)。

本論文 §1.3 指出，多篇研究觀察到真實 diffusion model 的生成結果中，關鍵特徵只在反向過程的一小段時間內出現；這些研究把它描述為 phase transition，稱為 critical window 或 speciation transition。

已驗證的相關工作：
- Georgiev et al. 2023, *The Journey, Not the Destination: How Data Guides Diffusion Models*（arXiv:2312.06205）。
- Raya & Ambrogioni 2023, *Spontaneous Symmetry Breaking in Generative Diffusion Models*（arXiv:2305.19693）。
- Biroli et al. 2024, *Dynamical Regimes of Diffusion Models*（arXiv:2402.18491）。
- Sclocchi et al. 2024, *A Phase Transition in Diffusion Models Reveals the Hierarchical Nature of Data*（arXiv:2402.16991）。
- Li & Chen 2024, *Critical windows: non-asymptotic theory for feature emergence in diffusion models*（arXiv:2403.01633）。

本論文另引用 Li, Karan & Chen 2025 等條目，arXiv ID 未能驗證，不列入。

與本論文的關係：本論文指出，這些工作研究 critical window 的現象，但「用 critical window 推導 diffusion sampling 的新 query 上界」是本論文首次提出。論文的分離結果即來自 masked diffusion 的 critical window 比 uniform、Gaussian 更窄，見 [本論文的分離結果](dllm-parallelism-separation-result.html)。
