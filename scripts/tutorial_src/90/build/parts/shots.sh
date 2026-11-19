#!/bin/sh
# screenshot helper: one chrome per shot, isolated profile
cd /tmp/claude-1000/-media-datatank-toonnyy8-toonnyy8-notes-github-io/5b278533-609c-4705-9f64-8b225c95d4e4/scratchpad/wt-paper-master/scripts/tutorial_src/90/build/caps
for spec in "1440,1400 c2-1440 #c2" "1440,1400 c3-1440 #c3" "1440,1400 c4-1440 #c4" "1440,1400 c5-1440 #c5" "390,1400 c1-390 #c1" "390,1400 c3-390 #c3" "390,1400 c4-390 #c4" "390,1400 c5-390 #c5" "390,1200 top-390 #top" "390,1400 results-390 #results"; do
  set -- $spec
  setsid timeout 40 google-chrome --headless=new --disable-gpu --no-sandbox --hide-scrollbars --force-prefers-reduced-motion --user-data-dir=$(mktemp -d) --virtual-time-budget=5000 --window-size=$1 --screenshot=/tmp/claude-1000/-media-datatank-toonnyy8-toonnyy8-notes-github-io/5b278533-609c-4705-9f64-8b225c95d4e4/scratchpad/wt-paper-master/scripts/tutorial_src/90/build/caps/$2.png "file:///tmp/claude-1000/-media-datatank-toonnyy8-toonnyy8-notes-github-io/5b278533-609c-4705-9f64-8b225c95d4e4/scratchpad/wt-paper-master/scripts/tutorial_src/90/build/index.html$3" >/dev/null 2>&1 </dev/null
done
setsid timeout 40 google-chrome --headless=new --disable-gpu --no-sandbox --hide-scrollbars --force-prefers-reduced-motion --force-dark-mode --user-data-dir=$(mktemp -d) --virtual-time-budget=5000 --window-size=1440,1400 --screenshot=/tmp/claude-1000/-media-datatank-toonnyy8-toonnyy8-notes-github-io/5b278533-609c-4705-9f64-8b225c95d4e4/scratchpad/wt-paper-master/scripts/tutorial_src/90/build/caps/dark-c4.png "file:///tmp/claude-1000/-media-datatank-toonnyy8-toonnyy8-notes-github-io/5b278533-609c-4705-9f64-8b225c95d4e4/scratchpad/wt-paper-master/scripts/tutorial_src/90/build/index.html#c4" >/dev/null 2>&1 </dev/null
echo done
