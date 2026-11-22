# Goldfish 驗收作答：No. 587 RIPO

## LO1
- **答**：$r_{s,a}(\theta)=\pi_\theta(a|s)/\pi_{\theta_{old}}(a|s)$，即更新中策略與行為策略對同一 token 的機率之比（重要性比值）。PPO-Clip 把 $r$ 限制在 $(1-\epsilon,1+\epsilon)$，等價於 $|r-1|<\epsilon$，常用 $\epsilon=0.2$。超出範圍時：$\hat A>0$ 且 $r>1+\epsilon$，或 $\hat A<0$ 且 $r<1-\epsilon$，該 token 梯度為 0，不再被推動。clip 並非硬性截斷，參數共享下比值仍可能超出範圍。
- **段落**：「這個 token 的梯度為 0，不再被推高」；「不是把 πθ 硬性截斷」
- **信心**：high

## LO2
- **答**：$\epsilon=0.2$、$\hat A>0$：$\pi_{old}=0.8$ 的上限為 0.96（增加 0.16）；$\pi_{old}=0.01$ 的上限為 0.012（增加 0.002）。DAPO（$\epsilon_{high}=0.28$）：$\pi_{old}=0.01$ 上限為 0.0128（增加 0.0008）；$\pi_{old}=0.8$ 的上限為 1（$0.8\times1.28=1.024$ 被機率上限 1 截斷）。
- **段落**：「原本機率 0.01 的動作只到 0.012」；「0.8 的上限則由 0.96 變成 1」
- **信心**：high

## LO3
- **答**：在 $\theta_{old}$ 附近，$D_{KL}(\pi_{\theta_{old}}\|\pi_\theta)\approx\tfrac12\Delta\theta^\top F(\theta)|_{\theta_{old}}\Delta\theta$，其中 $F=\mathbb E_{a\sim\pi_{\theta_{old}}}[\nabla_\theta\log\pi_\theta\,\nabla_\theta\log\pi_\theta^\top]$（Fisher 資訊矩陣），$\Delta\theta=\theta-\theta_{old}$。KL 在 $\theta_{old}$ 處的值與梯度為 0，只剩二次項。
- **段落**：「KL 的值與梯度都是 0，只剩二次項」
- **信心**：high

## LO4
- **答**：代入 $\Delta\pi\approx\nabla\pi^\top\Delta\theta$ 後，$(\Delta\pi)^2/\pi_{old}=\pi_{old}(r-1)^2$，所以 Eq. 7 中 $(r-1)^2$ 的權重是 $\pi_{old}$。意義：同樣的比值偏差 $r-1$，在高機率 token 上對應的距離大，在低機率 token 上對應的距離小，即幾何不均勻。
- **段落**：「同一個比值偏差，在高機率區對應的距離大」
- **信心**：high

## LO5
- **答**：PPO-Clip 隱含的是歐氏距離 $d_{clip}=(r-1)^2$（Eq. 3），與 $\pi_{old}$ 無關，等於假設策略變化在流形上均勻。以論文的單 token 項 $\tfrac12\pi_{old}(r-1)^2$、$|r-1|=0.2$ 計算：$\pi_{old}=0.8$ 為 $0.5\times0.8\times0.2^2=0.016$；$\pi_{old}=0.01$ 為 $0.5\times0.01\times0.2^2=0.0002$，相差 80 倍。少算的是：Eq. 7 對所有動作求和，其餘 token 因機率總和為 1 而跟著改變的部分。單 token 項只保留被取樣 token 那一項，因此低估分布層級的變化（頁面的推導示例中，二元量測在 $\pi_{old}=0.8$ 時約 400 倍、二元 KL 約 1000 倍差距）。
- **段落**：「這不是完整的 KL」；「Eq. 3」
- **信心**：high（數字）；medium（「少算了什麼」的二元量測細節屬頁面推導）

## LO6
- **答**：由 $\tfrac12\pi_{old}(r-1)^2\le\delta$ 得 $|r-1|\le\sqrt{2\delta/\pi_{old}}$（Eq. 10）。實作式 Eq. 11 為 $\epsilon=\sqrt{\delta/\pi_{old}}$ 沒有 2，因為係數 2 被吸收進超參數 $\delta$。因此以 Eq. 11 的 $\delta$ 而言，Eq. 9 單位下每個 token 的距離上限是 $\delta/2$。
- **段落**：「因為係數 2 被吸收進超參數 δ」
- **信心**：high

## LO7
- **答**：$\delta=0.02$：$\pi_{old}=0.8$ 的上限為 0.92（精確 0.9265），PPO 為 0.96；$\pi_{old}=0.01$ 的上限為 0.024（精確 0.0241），PPO 為 0.012。兩個 token 在上邊界的單 token 距離都是 0.01；PPO 的距離分別為 0.016 與 0.0002。
- **段落**：「兩個 token 在上邊界的單 token 距離都是 0.01」
- **信心**：high

## LO8
- **答**：RIC 的 $\epsilon$ 比 PPO 的 $\epsilon$ 小，條件是 $\sqrt{\delta/\pi_{old}}<\epsilon$，即 $\pi_{old}>\delta/\epsilon^2$。$\delta=0.02$、$\epsilon=0.2$ 時分界為 0.5。論文預設 $\delta=0.05$ 時分界為 1.25，因此對所有 $\pi_{old}$ 都有 $\epsilon_{s,a}\ge\sqrt{0.05}=0.224>0.2$，RIC 上界處處比 PPO 大（只是大的程度隨機率改變；$\pi_{old}\ge0.8$ 時有效上限受 $\pi\le1$ 限制而到 1.0）。
- **段落**：「RIC 的上界處處比 PPO 大」
- **信心**：high

## LO9
- **答**：clip 邊界上 $v=\pi_{old}w^2$（變異貢獻）：
  - 未截斷 IS：$v=\pi_\theta^2/\pi_{old}$，$\pi_{old}$ 很小時可任意大，偏差為 0。
  - PPO：$v=\pi_{old}(1+\epsilon)^2$，$\pi_{old}\to0$ 時趨近 0，但偏差大。
  - RIC：$v=\pi_{old}(1+\sqrt{\delta/\pi_{old}})^2=(\sqrt{\pi_{old}}+\sqrt\delta)^2$，$\pi_{old}\to0$ 時趨近 $\delta$（$O(\delta)$，與密度無關）。
  - 偏差定義：截斷把權重 $r$ 換成 $w$，期望少了 $\sum_x(\pi_\theta(x)-\pi_{old}(x)w(x))A(x)$；取 $A\equiv1$ 時，單一 token 的偏差是 $\pi_\theta-\pi_{old}w$。
- **段落**：「取 A≡1 時，單一 token 的偏差是」；「在 πold→0 時趨近 δ」
- **信心**：high

## LO10
- **答**：RIPO 只改兩處：(1) clip 範圍改為逐 token 的 $\epsilon_{i,t}=\sqrt{\delta/\pi_{old}}$（用舊機率，Eq. 11），取代固定 $\epsilon$；(2) 正規化由「先對每條回答平均、再對 G 條平均」改為 $1/\sum_i|o_i|$（token 層級，平衡長短回答）。預設 $\delta=0.05$；雙重 clipping 上下界為 0.5 與 10（所有方法都用）；移除 KL 懲罰。
- **段落**：「clip 範圍改為逐 token 的」；「lower and upper bounds set to 0.5 and 10」
- **信心**：high

## LO11
- **答**：Qwen3-8B-Base 七個數學基準平均（Avg@8）：RIPO 38.5、GRPO 28.5。δ 消融（AIME24，Table 2）：對稱 $\delta$ 從 0.02 到 0.08 為 40.8–43.8，表現穩定；$\delta_{low}/\delta_{high}\ge2$ 的三組降到 27.5–28.8，即不對稱會大幅傷害表現。Pass@128：AIME-25 RIPO 60.0 vs GRPO 53.3；HMMT-25 RIPO 45.3 vs GRPO 30.0。作者據此稱 RIPO 突破了基底模型的能力邊界（Base 在 k≈16 後不再上升）。
- **段落**：「Qwen3-8B-Base 38.5 vs 28.5」；「δ 消融（Table 2）」；「Pass@128」
- **信心**：high（數字）；medium（δ 消融每格只跑一次，無誤差範圍，頁面已說明）

## LO12
- **答**：RIPO 假設 $\pi_{old}$ 就是實際取樣的策略，修改的是 clip 上下界的形狀（逐 token 的 $\epsilon$）。Score Centering 處理推論引擎（sampler $q$）與訓練引擎（trainer $p$）機率不一致的情形，從每個 score 減去 sampler 下的期望 score，是 policy gradient 的 score 項上的加法修正。兩者作用位置不同，原則上可疊加，但兩篇論文都沒有組合實驗（疊加是頁面的推導）。
- **段落**：「修改的是 clip 上下界的形狀」；「兩篇也都沒有組合實驗」
- **信心**：high

## LO13
- **答**：W3 是頁面自行設計的玩具 bandit（非論文結果）。固定 $\epsilon=0.2$ 時，稀有動作 B 的比值超過 1.2 後就不再提供梯度，每次迭代只能小幅上升，策略先集中到 A，因此很少找到 B（GRPO 第 200 次迭代找到比例 0.015）。RIC 的增益不能歸因於「範圍形狀」：不 clip 對照為 0.395，RIC 為 0.370，相近；把 DAPO 的 $\epsilon_{high}$ 調到 1.0（下界保持 0.2）反而達 0.505。
- **段落**：「B 被取樣到後比值超過 1.2 就不再提供梯度」；「不能說明為什麼需要 trust region」
- **信心**：medium（玩具模型，頁面自行示意）

## 初學者最可能卡住的 5 處
1. **$\epsilon$ 與 $\delta$ 的係數 2**：Eq. 9 的距離上限是 $\delta/2$，Eq. 11 的 $\delta$ 已吸收 2。照 Eq. 9 理解會差一倍。
2. **「距離」與「比值範圍」不同**：PPO 的 $|r-1|<\epsilon$ 是比值範圍，$d_{clip}$ 與 $d_{geom}$ 是距離，兩者不能直接比。
3. **$\min$ 的符號**：$\hat A>0$ 只看上界、$\hat A<0$ 只看下界，頁面只在文字中說明，不易一眼看出。
4. **KL 推導（Eq. 4–7）**：需要 Fisher 矩陣、二階展開與 $\Delta\pi\approx\nabla\pi^\top\Delta\theta$ 的假設，頁面未先補背景（頁面另有連結的背景頁）。
5. **$\pi_{old}\to0$ 與雙重 clipping**：$\epsilon\to\infty$，雙重 clipping 的「上界 10」語意論文沒有說明；頁面中「本頁推導」「論文」標記混在一起，容易誤讀為論文結論。
