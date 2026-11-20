"""LO self-check: key terms of each reference answer (design.md section 2) must appear in accept/page.txt."""
import pathlib
t = (pathlib.Path(__file__).resolve().parents[2] / 'accept/page.txt').read_text()
checks = {
 'LO1': ['接收題目', '參考答案', '策略回答', '輸出 YES 或 NO', 'YES 給獎勵', 'NO 給'],
 'LO2': ['不含任何解答、卻常讓評審輸出 YES 的回答', '非文字符號', '推理開頭語', '「:」與「.」', '「Thought process:」與「Solution」'],
 'LO3': ['把回答固定為某個 key，在資料集全部題目上計算評審判 YES 的百分比', '10 個 key 的平均與最大', '50 格的平均與最大', '本頁推導'],
 'LO4': ['LLaMA3-70B-Instruct 80.6 | 95.1', 'LLaMA3-8B 77.7（論文印出 76.9）', 'Qwen2.5-72B 66.8', 'Master-RM-7B 0.1 | 2.9', 'GPT-4o 為 14.6', 'Claude-4 為 12.4'],
 'LO5': ['少於 30 token', 'KL 急升', '94.26%', 'Thought Process:', '約 90% 的準確率', '不同的分布'],
 'LO6': ['r_{att}=p\\cdot\\mathrm{TPR}+(1-p)\\,\\beta', '\\Delta=f-r_{att}>0', '\\pi(1-\\pi)\\Delta', '單調增加', '無 KL 懲罰、單一', '玩具的極限，不是論文結論', '借自 Table 1'],
 'LO7': ['抽 2 萬題', 'GPT-4o-mini', '只留第一句', '標 NO', '避開 10 個測試 key', '18 萬筆', '交叉熵', 'lr 5e-6、1 epoch'],
 'LO8': ['\\kappa=(p_o-p_e)/(1-p_e)', '隨機猜也會一致的部分', '0.10'],
 'LO9': ['沒有退化，部分情況提升', 'Master-RM-7B 0.91 / 0.90', 'Master-RM-32B 0.89 / 0.87', 'Master-RM-7B 94.45 / 84.40', 'Master-RM-32B 95.15 / 86.80', '0.55 與 2.0 百分點', '接近分布內'],
 'LO10': ['只在 Qwen2.5 系列量測', '非單調', '0.5B 最低但 κ 0.10', '1.5–3B 急升', '7–14B 最平衡', '32B/72B 再升', 'Claude-4', '−55.3 百分點', '−10.8 百分點', '−83% vs −86%', '不足以證明規模趨勢'],
 'LO11': ['不能可靠地降低 FPR', '依模型與領域而定', '12.6 升到 40.4（+27.8 百分點）', '87.6 升到 95.5', '54.0 降到 38.1', '不能只歸因於 CoT'],
 'LO12': ['大幅降低 FPR', '常歸零', '87.6 降到 0.0', '仍有 50.1', '數學任務可省略題目', '一般推理要小心'],
}
bad = 0
for k, v in checks.items():
    miss = [s for s in v if s not in t]
    bad += bool(miss)
    print(k, 'OK' if not miss else 'MISSING ' + str(miss))
print('LO self-check:', 12 - bad, '/ 12 OK')
