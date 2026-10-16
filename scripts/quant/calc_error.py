import os, numpy as np

golden_dir = '/mnt/g/WSL/analysis_result/golden'
sim_dir = '/mnt/g/WSL/analysis_result/simulator'

files = sorted(os.listdir(golden_dir))
m23 = [f for f in files if 'model_23' in f and f.endswith('.txt')]
# 也看最终输出
out_files = [f for f in files if 'output0' in f.lower() or f in ('output0.txt',)]

def cos(a, b):
    a, b = a.flatten(), b.flatten()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 1.0 if na < 1e-9 and nb < 1e-9 else 0.0
    return float(np.dot(a, b) / (na * nb))

print(f'model.23 层: {len(m23)}, 输出层: {out_files}')
print(f'{"layer":<58} {"cos":>7} {"gold_max":>10} {"sim_max":>10} {"gold_mean":>10}')
print('-' * 100)
for f in m23 + out_files:
    g = np.loadtxt(os.path.join(golden_dir, f))
    s_path = os.path.join(sim_dir, f)
    if not os.path.exists(s_path):
        print(f'{f:<58} [sim missing]')
        continue
    s = np.loadtxt(s_path)
    gs, ss = g.flatten(), s.flatten()
    if gs.shape != ss.shape:
        print(f'{f:<58} [shape! g={g.shape} s={s.shape}] gmax={g.max():.4f} smax={s.max():.4f} <=== CHECK')
        continue
    c = cos(g, s)
    flag = ''
    if c < 0.95: flag = ' <=== LOW COS'
    if 'Sigmoid' in f or 'Concat_7' in f or 'output0' in f: flag += ' [KEY]'
    print(f'{f:<58} {c:>7.4f} {g.max():>10.4f} {s.max():>10.4f} {g.mean():>10.4f}{flag}')
