# Goldfish 驗收評分

**分數：12/12**（全部 correct，無 partial / wrong）

## 逐 LO
- LO1 correct：接收 q、a*、o，輸出 YES/NO，R=1/0，與參考一致。
- LO2 correct：定義與兩類各舉例齊全。
- LO3 correct：固定 key、全題計 YES%、Average/Worst=10 格、Overall=50 格並註明本頁推導。
- LO4 correct：80.6|95.1、77.7（†76.9）、66.8、0.1|2.9、GPT-4o 14.6、Claude-4 12.4 全對。
- LO5 correct：<30 token、KL 急升、Thought Process 94.26%、約 90% 且註明分布不同。
- LO6 correct：f、r_att、Δ>0、梯度 π(1-π)Δ、極限 π→1 與玩具限定均有。
- LO7 correct：2 萬題、GPT-4o-mini、只留首句標 NO、避開 key、18 萬、SFT 設定皆對。
- LO8 correct：κ 公式、扣除隨機一致、0.5B κ=0.10。
- LO9 correct：兩個 Master-RM 的 κ 與 VerifyBench 數字、與 GPT-o1 差距、近分布內皆對。
- LO10 correct：非單調、作者假設地位、−55.3 vs −10.8、−83% vs −86%、兩點不足。
- LO11 correct：不能；12.6→40.4、GSM8K 升而 AIME 降、CoT 欄三項混雜。
- LO12 correct：87.6→0.0、NaturalReasoning 50.1、適用範圍建議。

## 卡點檢視
五項卡點都已被頁面文字解開（讀者能引到對應句），屬「頁面已處理」而非缺陷：FPR 範圍、Average/Worst/Overall 為本頁推導、百分點 vs %、玩具非論文實驗、CoT 欄三項混雜。

## 修正清單（皆為小事，非必要）
1. LO9 段落（page.txt 約第 335 行附近）未標註 App. B.2，讀者自己指出依據只有 Sec. 3.3.2 / Table 2。建議在該段末加「（Table 2、App. B.2）」，並一句話說明 κ 評估回答由 Qwen2.5-7B-Instruct 產生、接近分布內（讀者已從頁面讀到，僅補出處）。
2. 其餘無需修改；LLaMA3-8B 的 59.0 vs 55.0 不一致（第 130 行）已清楚，維持現狀。
