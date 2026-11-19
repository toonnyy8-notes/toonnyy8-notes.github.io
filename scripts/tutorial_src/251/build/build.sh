#!/bin/sh
# Assemble build/index.html from parts; core.js, table1.json and timeline.json are embedded verbatim.
set -e
D=$(cd "$(dirname "$0")/.." && pwd); P=$D/build/parts
{
cat $P/head.html $P/shell.css $P/page.css; echo '</style>'; echo '</head>'
cat $P/body1.html $P/body2.html $P/body3.html
printf '<script type="application/json" id="t1-data">'; cat $D/verify/table1.json; echo '</script>'
printf '<script type="application/json" id="timeline-data">'; cat $D/notes/timeline.json; echo '</script>'
echo '<script>'; echo '(()=>{'; echo "'use strict';"
cat $P/helpers.js
grep -v '^if(typeof module' $D/verify/core.js
cat $P/tocTheme.js $P/app.js $P/boot.js
} > $D/build/index.html
