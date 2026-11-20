t=open('scripts/tutorial_src/441/accept/page.txt').read()
checks={'LO1':['致命三元組','off-policy','bootstrapping','d^{\\pi_\\phi}','D(d^{\\pi_\\phi}'],
'LO2':['\\log p_\\theta(x_0(\\tau)\\mid y(\\tau))','報酬','限制','技能','狀態序列'],
'LO3':['最佳（訓練完美）','邊際分布','-\\sqrt{1-\\bar\\alpha_k}','∝','App. D'],
'LO4':['兩個理由','離散','111.8、79.3、100','111.3、66.3、76.5','±1.6'],
'LO5':['p=0.25','Ø','\\omega=0','\\omega=1','1.2、1.4、1.6、1.8'],
'LO6':['\\omega\\log p_k(y\\mid x)','隱式分類器','不是同一個分布加噪'],
'LO7':['兩處','\\alpha I','104.3','111.8','107.1','\\alpha=0.5'],
'LO8':['C=20','1.26 s','0.48 s','Hopper-Medium-Expert','滾動視界'],
'LO9':['給定狀態軌跡 $x_0(\\tau)$','給定 $x_k(\\tau)$','可行','不支援 OR','App. I.2'],
'LO10':['81.8','77.6','78.9','75.3','DD 61','51.2','4 個最高','95.4','79.3'],
'LO11':['60.4','52.3','63.8','62.7','58.9','分布外','Bound 60.1%','Pace 38.5%','Trot 1.4%'],
'LO12':['77.3','63.2','61.8','p\\ge0.10','無法區分','OR','過擬合','完全可觀察']}
for k,v in checks.items():
    miss=[s for s in v if s not in t]; print(k,'OK' if not miss else 'MISSING '+str(miss))
