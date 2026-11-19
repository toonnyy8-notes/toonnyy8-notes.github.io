import pathlib, re
d = pathlib.Path(__file__).resolve().parent.parent
K = '/tmp/claude-1000/-media-datatank-toonnyy8-toonnyy8-notes-github-io/5b278533-609c-4705-9f64-8b225c95d4e4/scratchpad/katex/0.16.9'
s = (d / 'index.html').read_text().replace('https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9', 'file://' + K)
s = re.sub(r'<link[^>]*fonts\.(googleapis|gstatic)[^>]*>\n?', '', s)
inj = '<style>html{scroll-behavior:auto!important}</style><script>addEventListener("load",()=>setTimeout(()=>{const e=location.hash&&document.querySelector(location.hash);if(!e)return;document.querySelectorAll("main>section").forEach(x=>{if(!x.contains(e))x.style.display="none"});if(e.classList.contains("plate")){const sec=e.closest("section");[...sec.children].forEach(c=>{if(c!==e)c.style.display="none"})}},0))</script></body>'
(d / '.local.html').write_text(s.replace('</body>', inj))
