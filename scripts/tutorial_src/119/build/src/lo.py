"""LO self-check: key terms of each reference answer (design.md section 2) must appear in accept/page.txt."""
import pathlib
t = (pathlib.Path(__file__).resolve().parents[2] / 'accept/page.txt').read_text()
checks = {
 'LO1': ['只看狀態', '自己取樣軌跡', '設計者的固定規則產生', '每個 prefix', 'token 層級 divergence', '老師一側 stop-gradient（Eq. 2–3）', '推論時只用學生'],
 'LO2': ['老師與學生不是同一組權重', '凍結的 base checkpoint', '同架構、同初始化', 'encoder LoRA 與 QFormer', 'activations 留在計算圖中', 'Eq. 10 沒有對老師 stop-gradient', '梯度穿過凍結網路', 'encoder LoRA 在老師主 forward 時關閉'],
 'LO3': ['只存訓練集的成功 rollout', 'Qwen3-Embedding-8B（4,096 維）', 'FAISS 取 cosine 最高的', 'LoRA（rank 8）', '8 層共用參數的 QFormer', '96 個 latent token', '96 個 pad 佔位', 'torch.cat', 'Eq. 14 的 NLL 冷啟動'],
 'LO4': ['老師機率最高的', 'M=20', '合併成 ⊥', '學生用同一支撐集（Eq. 9）', '學生在前（reverse KL）', '以 ω 對動作 token 平均（Eq. 10）', '截斷 KL 不大於完整 KL（log-sum 不等式）', '只有尾部總量受監督'],
 'LO5': ['老師一側沒有 stop-gradient', 'composer 可以改變老師分布來降低損失', '移向學生', '梯度本身不含脈絡是否有資訊的訊號', '沒有資訊的脈絡'],
 'LO6': ['學生取樣的 token（Eq. 11）', 'A=2r-1', 'Eq. 12', '期望上不為正', '取樣平均可以為正', '溫度 0.7、top-p、top-k 取樣', '選了 token', '共變項', '第一項本身為正'],
 'LO7': ['β 初值 0', 'm=0.05', '\\eta_\\beta=0.5', '\\lambda=0.2', 'β 降到 0 為止', 'REINFORCE 形式', '無重要性權重', 'β 每步只增加 0.025'],
 'LO8': ['q^*(a)=p(a)(1+\\beta Q(a))/(1+\\beta\\bar Q)', '1+\\beta Q>0', 'β = 0 時', '（退化）', 'Q 對所有 token 相同時不論 β 多大', '學生固定、單一 prefix、老師 logits 自由', '不是論文結果'],
 'LO9': ['Frozen φ₀ 0.573', 'm=0 0.551', '0.01 0.566', '0.02 0.603', '0.05 0.637', '0.10 0.626', '0.20 0.613', '低於以凍結 composer 訓練的學生', '是作者認為', 'base＋φ₀', '單次訓練'],
 'LO10': ['10 組 backbone × benchmark 的彙總欄', 'Long Ctx 26.00', 'ACE M-Step 56.6', 'LCB v5 141 vs 140／279', 'v6 59 vs 58／131', 'HumanEval+ 144 vs 143／164', 'seed 數、標準差或信賴區間', '評測溫度 0.7', '依 EnvScaler 測試集選定', 'm 的選法未說明'],
 'LO11': ['320 generations 後超過 0.61', '576 時 0.637', '0.611、0.588', '320/1,600=20%', '576/1,600=36%', '對 Skill-SD 成立', '不知是否超過 0.611', '論文未說預算是否計入', '定期在測試集上的評測', 'GRPO 61.8、Skill-SD 59.1', '本頁不重畫'],
 'LO12': ['以文字放進老師脈絡', '不是純檢索', 'retrieval alone is insufficient', '0.631 與學生 0.637 相近', 'follow Zhao et al.（OPSD）', 'SDPO-style', '全詞表 forward KL'],
}
bad = 0
for k, v in checks.items():
    miss = [s for s in v if s not in t]
    bad += bool(miss)
    print(k, 'OK' if not miss else 'MISSING ' + str(miss))
print('LO self-check:', 12 - bad, '/ 12 OK')
