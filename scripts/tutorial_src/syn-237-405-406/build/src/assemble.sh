#!/bin/sh
set -e
cd "$(dirname "$0")"
R=../../../../..
H=$R/tutorials/exposure-bias-epsilon-scaling.html
{
sed -n '1,382p' $H | sed -e 's#<title>[^<]*</title>#<title>分數誤差、曝光偏差與三種縮放</title>#' \
 -e 's#<meta name="description" content="[^"]*">#<meta name="description" content="綜合教學：比較 Tempered Score（No. 237）、DDPM-IP（No. 405）與 Epsilon Scaling（No. 406）。三篇都處理取樣器需要的分數與網路實際給出的值之間的差；本頁以統一的誤差單位、分數乘數軸、平穩分佈與擾動強度對照，說明它們在哪裡相連、哪裡不相連。">#'
cat extra.css
echo '</style>'
echo '</head>'
cat body.html
echo '<script>'
echo "(()=>{"
echo "'use strict';"
grep -v -e "^'use strict';" -e "module.exports" ../../../237/verify/core.js
grep -v -e "^'use strict';" -e "module.exports" ../../verify/core.js
cat app.js
echo "})();"
echo '</script>'
echo '</body>'
echo '</html>'
} > ../index.html
