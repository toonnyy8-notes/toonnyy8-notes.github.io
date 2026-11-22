# consistency checks of paper numbers (Table 1/3/5/6, Sec 5.1)
T1={'Qwen3-1.7B-Base':(11.2,15.4,37.2,(11.3,18.3)),'Llama3.2-3B-Instruct':(6.4,8.6,34.4,(16.3,22.1)),
    'Qwen3-4B-Base':(25.7,30.1,17.1,(30.4,32.9)),'Qwen3-8B-Base':(28.5,38.5,35.1,(31.7,43.8))}
for k,(g,r,rep,(ga,ra)) in T1.items():
    print(f"{k}: avg rel {100*(r/g-1):.1f}% (paper {rep}%), AIME24 rel {100*(ra/ga-1):.1f}%")
print('Sec5.1 batch: 128 prompts x 8 =',128*8,'; minibatch 16x8=',16*8,'; updates/iter',1024//128)
print('Sec5.6: 7K/256*15 =',7000/256*15,' 7K/512*15 =',7000/512*15,' (paper: 435 steps)')
print('Table6 coding rel',round(100*(44.9/39.7-1),1),'search',round(100*(43.4/37.7-1),1))
