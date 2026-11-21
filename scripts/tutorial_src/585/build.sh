#!/bin/sh
# assemble build/index.html from build/parts (run from anywhere)
D=$(cd "$(dirname "$0")" && pwd); P=$D/build/parts; O=$D/build/index.html
{
 echo '<!DOCTYPE html>'; echo '<html lang="zh-Hant-TW">'; echo '<head>'
 sed -n 1,2p $P/head.html
 echo '<title>HiT 階層擴散 Transformer</title>'
 echo '<meta name="description" content="圖解 HiT：在像素空間擴散 Transformer 中以親和度分組、軟池化與批次層級壓縮預算學出視覺階層，並以群組引導取樣。">'
 sed -n '3,$p' $P/head.html
 cat $P/css.html $P/css2.html; echo '</style>'
 cat $P/body1.html $P/body2.html
 printf '<script type="application/json" id="timeline-data">'; python3 -c "import json,sys;print(json.dumps(json.load(open(sys.argv[1])),ensure_ascii=False),end='')" $D/notes/timeline.json; echo '</script>'
 printf '<script type="application/json" id="c2-data">'; cat $D/verify/c2_scores.json; echo '</script>'
 echo '<script>'; cat $P/script.js; echo '</script>'; echo '</body>'; echo '</html>'
} > $O
echo built $(wc -c < $O) bytes
