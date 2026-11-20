#!/bin/sh
# usage: [DARK=1] ./caps.sh WIDTH HEIGHT id...  -> caps/<id>-<W>[-dark].png (shows only element #id)
D=$(cd "$(dirname "$0")" && pwd); W=$1; H=$2; shift 2
for id in "$@"; do suf=""; extra=""; [ -n "$DARK" ] && { suf="-dark"; extra="--force-dark-mode"; }
sed "s#</head>#<style>.appbar,.tocbar,.toc,\#pm-bar,footer{display:none!important}main>section:not(:has(\#$id)){display:none!important}main>section:has(\#$id)>*:not(\#$id):not(:has(\#$id)){display:none!important}section.ch{padding-top:8px!important}.shell{grid-template-columns:200px 1fr!important}.toc{display:block!important;visibility:hidden}</style></head>#" "$D/.t.html" > "$D/.c.html"
timeout 60 google-chrome --headless=new --disable-gpu --no-sandbox --hide-scrollbars $extra --virtual-time-budget=6000 --window-size=$W,$H --screenshot="$D/caps/$id-$W$suf.png" "file://$D/.c.html" 2>/dev/null >/dev/null; done
