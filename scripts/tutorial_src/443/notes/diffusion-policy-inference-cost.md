---
slug: diffusion-policy-inference-cost
title: 擴散策略的推論成本與論文提出的加速方向
cluster: RL, Reasoning & Self-Distillation
kind: topic
tags: [inference-cost, sampling-speed, ddim, dpm-solver, distillation, diffusion-policy]
arxiv: ["2208.06193", "2010.02502", "2206.00927", "2202.00512", "2112.07804"]
sources: ["paper.txt §4, §5, Appendix G", "arXiv API 驗證 2010.02502、2206.00927、2202.00512、2112.07804 的題名與作者；這些工作的內容未另讀，僅記錄論文把它們列為可能的加速方法"]
related: ["diffusion-policy-class", "fewer-steps-sampling-acceleration", "ddpm-quality-and-ddim-sampling", "diffusion-policy-experiments-d4rl"]
---
# 擴散策略的推論成本

[擴散模型作為策略類別](diffusion-policy-class.html) 中，每選一個動作需執行 `N` 次 `ε_θ` 網路。論文（arXiv 2208.06193）的相關敘述：

- bandit 實驗：`N` 大則更貼近資料分布但計算成本高；`N` 是表達力與計算成本的取捨。純 BC 擴散在 `N=50` 時能準確還原資料分布。
- D4RL：取 `N=5`。Gym 任務的 `N∈[2,5,10,20]` 消融中，`N=5` 已足夠；`N=5` 時訓練時間與 CQL 相近。
- 限制（Appendix G）：`N=5` 的擴散策略在動作推論上可能比單步前饋策略慢四到五倍，對需要快速反應的真實部署是障礙。

## 論文列出的加速方向

論文在未來工作中提出：可套用加速擴散取樣的相關研究，並設想訓練後把擴散策略蒸餾成較簡單的前饋策略。引用的相關工作（皆已以 arXiv API 驗證題名）：

- DDIM（Song, Meng, Ermon，2020，arXiv 2010.02502）
- DPM-Solver（Lu et al.，2022，arXiv 2206.00927）
- Progressive Distillation（Salimans & Ho，2022，arXiv 2202.00512）
- Denoising Diffusion GANs（Xiao et al.，2021，arXiv 2112.07804）

另有 Wang et al.（2022）Diffusion-GAN 與 Zheng et al.（2022）截斷擴散模型，本批次未驗證其 arXiv ID，不列入時間線。通用的加速背景見 [減少取樣步數與加速訓練的相關工作](fewer-steps-sampling-acceleration.html) 與 [DDPM 品質與 DDIM 取樣](ddpm-quality-and-ddim-sampling.html)。
