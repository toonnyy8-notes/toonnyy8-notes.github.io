"""C2 acceptance: partition metrics on the toy scenes."""
import sys, time
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
eps = noise_field()
for name in ('landscape',):
    img = scene(name); det = detail(img)
    print(f"## {name}: detail share top15%={detail_share(det,.15):.3f} top50%={detail_share(det,.5):.3f}")
    for t in (1.0, 0.5, 0.0) if name == 'landscape' else (1.0,):
        cur = img if t == 1.0 else mix(img, eps, t)
        h = patch_features(cur)
        for R in (64, 96, 128, 256, 512):
            out = []
            for meth in ('adaptive', 'fixed', 'featsim'):
                lab = adaptive(h, R) if meth == 'adaptive' else fixed(R) if meth == 'fixed' else featsim(h, R)
                e = ev(h, lab); st = group_stats(lab, det)
                rt = '—' if st['ratio'] is None else f"{st['ratio']:.2f}"
                out.append(f"{meth}: EV={e:.3f} spr={st['spread']:.3f} corr={st['corr']:+.3f} G={st['groups']} max={st['maxsize']} ratio={rt}")
            out.append(f"skip: EV={ev_skip(h,R):.3f}")
            print(f"t={t} R={R} | " + " | ".join(out))
