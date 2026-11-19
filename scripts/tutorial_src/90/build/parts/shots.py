"""usage: python3 -I shots.py <caps dir> <name:WxH:query[:dark]> ...  one isolated headless chrome per shot."""
import sys, subprocess, os, signal, tempfile, pathlib
C = pathlib.Path(sys.argv[1])
for spec in sys.argv[2:]:
    name, size, q, *flags = spec.split(':')
    args = ['google-chrome', '--headless=new', '--disable-gpu', '--no-sandbox', '--hide-scrollbars', '--allow-file-access-from-files',
            '--virtual-time-budget=6000',
            f'--window-size={size.replace("x", ",")}', f'--screenshot={C}/{name}.png']
    if 'dark' in flags: args.append('--force-dark-mode')
    url = f'file://{C}/shot.html?{q}' if q != 'page' else f'file://{C}/local.html'
    p = subprocess.Popen(args + [url], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    try: p.wait(timeout=45)
    except subprocess.TimeoutExpired: pass
    try: os.killpg(p.pid, signal.SIGKILL)
    except ProcessLookupError: pass
    print(name, (C / f'{name}.png').exists(), flush=True)
