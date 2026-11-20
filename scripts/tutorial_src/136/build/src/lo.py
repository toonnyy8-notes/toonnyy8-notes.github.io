"""LO self-check: key terms of each reference answer (design.md section 2) must appear in accept/page.txt."""
import pathlib
t = (pathlib.Path(__file__).resolve().parents[2] / 'accept/page.txt').read_text()
checks = {
 'LO1': ['OffPD 取', '$\\rho=\\pi_T$', 'OnPD 取', '$\\rho=\\pi_S^\\theta$', 'stop-gradient', '半梯度（semi-gradient，Sec. 3、Eq. 19）'],
 'LO2': ['以教師機率加權', 'mode-covering', '以學生機率加權', 'mode-seeking'],
 'LO3': ['鏈式分解（Prop. C.4，條件是固定生成長度', '只在目標值層級成立', '不偏估計', '本文在全詞彙上算 KL，KL 方向不限制 rollout', '不代表慣例配對較容易最佳化或表現較好'],
 'LO4': ['rollout（OnPD／OffPD）、KL 方向（forward／reverse）、學習率（1×10⁻⁵／5×10⁻⁵）', 'Llama-3.1-8B', '150 步', '7 個 OOD 基準平均分數', '$\\tau=10^{-6}$'],
 'LO5': ['OnPD 72%、OffPD 73%', '71–73%', '35%–72%', '最多變動 1.3 個百分點', '11.2–14.0 個百分點', '85.3–89.6%', '51.9–60.0%', '至少與 OnPD 一樣稀疏'],
 'LO6': ['$\\pi_S(v)-\\pi_T(v)$', '$[-1,1]$', '學生機率趨近 0 時這個座標消失', '難以找回學生遺漏的教師 mode', '可任意大'],
 'LO7': ['Eq. 26', '$\\le2\\sqrt2B\\,\\mathbb E_x[\\mathrm{TV}]$', '完整輸出序列的分布距離', '\\frac{\\varepsilon\\sqrt2}8\\log\\frac{1-\\delta}\\delta', '不存在與分布無關的常數', '$2BR\\,\\mathbb E_x[\\mathrm{TV}]$'],
 'LO8': ['不能推出', '一步的期望半梯度', 'Jacobian 上界', '也不解釋遺忘與稀疏度', '只能解讀為與定理一致'],
 'LO9': ['時就是', '（OffPD）', '（OnPD）', '正規化後的幾何中點', '$\\alpha=0.2$', '$c=2\\log3$', '最多 3 倍'],
 'LO10': ['都高於 80%', '只變動 5.2 個百分點', '受益於學生偏好的 rollout', '偶爾準確率驟降', '主要由學習率決定'],
 'LO11': ['Countdown-4E', '一致高 10–15%', '未說明是百分點或相對比例', '0.0%／0.1%，每條件 1 次訓練', '可能有助於抑制', '最強的持續表現來自 off-policy checkpoint'],
 'LO12': ['抽樣 KL：forward 仍穩健、reverse 仍敏感', 'OffPD 尤其明顯', '沒有一致的 OnPD–OffPD 差距', 'Fig. 8 沒有數字標籤', '622 tokens', '59.0%', '偏樂觀', '每條件只有 1 次訓練'],
}
bad = 0
for k, v in checks.items():
    miss = [s for s in v if s not in t]
    bad += bool(miss)
    print(k, 'OK' if not miss else 'MISSING ' + str(miss))
print('LO self-check:', 12 - bad, '/ 12 OK')
