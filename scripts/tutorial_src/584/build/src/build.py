import json,pathlib
here=pathlib.Path(__file__).parent; root=here.parent.parent.parent.parent.parent
ref=(root/'tutorials/prediction-targets-k-diff.html').read_text().splitlines()
css='\n'.join(ref[14:165]+ref[183:340])
helpers='\n'.join(ref[910:939])
helpers=helpers.replace("const RM=matchMedia('(prefers-reduced-motion: reduce)').matches;\n","")
tl=json.dumps(json.load(open(here.parent.parent/'notes/timeline.json')),ensure_ascii=False)
head='''<!DOCTYPE html>
<html lang="zh-Hant-TW">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Attention Sink 與原生 MoE</title>
<meta name="description" content="圖解 Fu 等人的 ICML 2026 論文：attention sink 的權重 1 − A_sink 是隱式門控，讓注意力層成為原生 MoE；以 sink-aware 負載平衡損失處理 head collapse。">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Newsreader:opsz,wght@6..72,400;6..72,500;6..72,600&family=Noto+Serif+TC:wght@600;700&family=IBM+Plex+Sans:wght@400;500;600&family=Noto+Sans+TC:wght@400;500;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/katex.min.css">
<script defer src="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/katex.min.js"></script>
<script defer src="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/contrib/auto-render.min.js"></script>
<style>
'''
out=head+css+'\n'+(here/'extra.css').read_text()+'</style>\n</head>\n'+(here/'body.html').read_text()+'\n<script type="application/json" id="timeline-data">'+tl+'</script>\n<script>\n'+helpers+'\n'+(here/'app.js').read_text()+'})();\n</script>\n</body>\n</html>\n'
(here.parent/'index.html').write_text(out); print(len(out))
