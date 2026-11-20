# usage: .shot.sh <anchor> <width> <height> <out> [extra chrome flags]
a=$1; w=$2; h=$3; out=$4; shift 4
echo "<html><body style=margin:0><iframe id=f src=\".t.html\" width=$w height=$h style=border:0></iframe><script>f.onload=()=>setTimeout(()=>{const e=f.contentDocument.getElementById('$a');if(e)f.contentWindow.scrollTo(0,e.getBoundingClientRect().top+f.contentWindow.scrollY-70)},2500)</script></body></html>" > .f_shot.html
timeout 60 google-chrome --headless=new --disable-gpu --no-sandbox --hide-scrollbars --allow-file-access-from-files --virtual-time-budget=8000 --window-size=$w,$h "$@" --screenshot=$out "file://$PWD/.f_shot.html" >/dev/null 2>&1
