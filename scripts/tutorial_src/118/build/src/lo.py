"""LO self-check: key terms of each reference answer (design.md section 2) must appear in accept/page.txt."""
import pathlib
t = (pathlib.Path(__file__).resolve().parents[2] / 'accept/page.txt').read_text()
checks = {
 'LO1': ['選最高分的一份', '等於 Pass@1', '等於 Pass@N', '1-(1-p)^N', '各題這個值的平均'],
 'LO2': ['RM accuracy 只看單份解答', '逐點', '排序', '兩者相關但不相同'],
 'LO3': ['r_\\theta=\\mathrm{sigmoid}(z_{cls})', 'BCE', '各半', '不產生文字', '不能先寫 CoT', '同一個輸出頭上統一'],
 'LO4': ['r_{\\text{Direct}}=p_\\theta(\\text{Yes}\\mid\\mathbf x,\\mathbf y,I)', 'Is the answer correct (Yes/No)?', '正確解→Yes、錯誤解→No', '\\mathrm{sigmoid}(z_Y-z_N)', 'BCE 形式相同', '論文沒有量測', '相當或略好（Fig. D.1）'],
 'LO5': ['\\lambda\\mathcal L_{\\text{SFT}}(\\mathcal D_{\\text{correct}})', '演算法任務 1/3、GSM8K 1/4', 'Fig. 9', '過多會降低驗證表現（Fig. D.3）', 'Pass@N 也提高（Fig. 10）'],
 'LO6': ['Let\'s verify step by step.', '生成驗證理由', '再接上 I $I$', 'Yes 的機率為分數（Eq. 6）'],
 'LO7': ['不是票數', 'K=32', '做不到', '〔本頁說明〕', '確定的分數', 'K=2', 'greedy'],
 'LO8': ['Gemini 1.0 Pro', '參考解答', 'Table A.2', '答案檢查器一致', '超過 50%', '推論時都不放入', '91.7%', '87.8%'],
 'LO9': ['5.0% → 45.3%', '73.0% → 93.4%', '28.0% → 44.6%', '37.9% → 53.5%', '16.6 個百分點', '17% 是百分點', '50% → 53.5%'],
 'LO10': ['2B → 7B → 9B', '每個大小都較好（Fig. 12）', 'Gemini 1.0 Pro LLM-as-a-Judge（Fig. 11）', '6.4 倍'],
 'LO11': ['對每個最終答案加總', 'K_{\\text{top}}', 'GSM8K 用 6、MATH 用 4', '2.5 倍', '93.4% 到 93.5%'],
 'LO12': ['正確、錯誤各半', '驗證理由', '生成資料（Eq. 5）', 'Yes token', '專用 cls $cls$ 的 logit', 'next-token 的 SFT', '平均 K 條理由'],
}
bad = 0
for k, v in checks.items():
    miss = [s for s in v if s not in t]
    bad += bool(miss)
    print(k, 'OK' if not miss else 'MISSING ' + str(miss))
print('LO self-check:', 12 - bad, '/ 12 OK')
