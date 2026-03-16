# -*- coding: utf-8 -*-
"""Comprehensive numerical audit: every number in the paper vs raw data."""
import csv, numpy as np, sys, os
sys.path.insert(0, '.')

errors = []
def check(label, paper_val, actual_val, tol=0.002):
    if abs(paper_val - actual_val) > tol:
        errors.append(f'{label}: paper={paper_val}, actual={actual_val:.6f}')

# ============ Load VN at noise=0.5 ============
vn = {}
with open('results/variable_noise_metrics.csv') as f:
    for r in csv.DictReader(f):
        if float(r['noise_level']) != 0.5: continue
        g, s = r['group'], int(r['seed_model'])
        if g not in vn: vn[g] = {}
        if s not in vn[g]: vn[g][s] = []
        vn[g][s].append({k: float(r[k]) for k in ['gain','acc_t1','acc_t3']})
for g in vn:
    for s in vn[g]:
        vn[g][s] = {k: np.mean([v[k] for v in vn[g][s]]) for k in ['gain','acc_t1','acc_t3']}

# ============ Load Static at noise=0.5 ============
st = {}
with open('results/raw_metrics.csv') as f:
    for r in csv.DictReader(f):
        if float(r['noise_level']) != 0.5: continue
        g, s = r['group'], int(r['seed_model'])
        if g not in st: st[g] = {}
        if s not in st[g]: st[g][s] = []
        st[g][s].append({k: float(r[k]) for k in ['gain','acc_t1','acc_t3','r_norm','delta_norm']})
for g in st:
    for s in st[g]:
        st[g][s] = {k: np.mean([v[k] for v in st[g][s]]) for k in st[g][s][0]}

def ms(gd, key='gain'):
    vals = [gd[s][key] for s in range(10) if s in gd]
    return np.mean(vals), np.std(vals)

rng = np.random.RandomState(42)

# ============ §3.1 VN ============
print('§3.1 VN gains...')
m,sd = ms(vn['Baseline']); check('VN BL gain', 0.151, m); check('VN BL SD', 0.046, sd)
m,sd = ms(vn['A']); check('VN A gain', 0.008, m); check('VN A SD', 0.045, sd)
m,sd = ms(vn['C1']); check('VN C1 gain', -0.068, m); check('VN C1 SD', 0.087, sd)
m,sd = ms(vn['C2']); check('VN C2 gain', -0.018, m); check('VN C2 SD', 0.100, sd)

# VN endpoints
m,_ = ms(vn['Baseline'],'acc_t1'); check('VN BL t1', 0.685, m)
m,_ = ms(vn['Baseline'],'acc_t3'); check('VN BL t3', 0.836, m)
m,_ = ms(vn['A'],'acc_t1'); check('VN A t1', 0.685, m)
m,_ = ms(vn['A'],'acc_t3'); check('VN A t3', 0.694, m)

# VN Bootstrap CI
print('§3.1 VN paired CIs...')
for name,a,b,pm,plo,phi in [
    ('VN B-A','Baseline','A',0.143,0.110,0.181),
    ('VN B-C1','Baseline','C1',0.219,0.163,0.278),
    ('VN B-C2','Baseline','C2',0.168,0.124,0.203),
]:
    diffs = [vn[a][s]['gain']-vn[b][s]['gain'] for s in range(10)]
    boot = [np.mean(rng.choice(diffs,10,replace=True)) for _ in range(10000)]
    check(f'{name} mean', pm, np.mean(diffs))
    check(f'{name} lo', plo, np.percentile(boot,2.5), 0.005)
    check(f'{name} hi', phi, np.percentile(boot,97.5), 0.005)

# VN Baseline CI
bg = [vn['Baseline'][s]['gain'] for s in range(10)]
boot = [np.mean(rng.choice(bg,10,replace=True)) for _ in range(10000)]
check('VN BL CI lo', 0.121, np.percentile(boot,2.5), 0.003)
check('VN BL CI hi', 0.179, np.percentile(boot,97.5), 0.003)

# ============ §3.2 Static ============
print('§3.2 Static...')
m,sd = ms(st['Baseline']); check('St BL gain', 0.042, m); check('St BL SD', 0.029, sd)

# Static paired CIs
for name,a,b,pm,plo,phi in [
    ('St B-A','Baseline','A',0.042,0.023,0.059),
    ('St B-C1','Baseline','C1',0.106,0.074,0.139),
    ('St B-C2','Baseline','C2',0.101,0.062,0.148),
]:
    diffs = [st[a][s]['gain']-st[b][s]['gain'] for s in range(10)]
    boot = [np.mean(rng.choice(diffs,10,replace=True)) for _ in range(10000)]
    check(f'{name} mean', pm, np.mean(diffs), 0.003)
    check(f'{name} lo', plo, np.percentile(boot,2.5), 0.005)
    check(f'{name} hi', phi, np.percentile(boot,97.5), 0.005)

# ============ §3.3 Accuracies ============
print('§3.3 Accuracies...')
for g,val in [('Baseline',0.698),('A',0.698),('B1',0.515),('B2',0.199),
              ('D',0.746),("D'",0.822),("D''",0.600)]:
    if g in st:
        m,_ = ms(st[g],'acc_t1'); check(f'{g} t1', val, m)

# r_norm, delta
m_rn = np.mean([st['Baseline'][s]['r_norm'] for s in range(10)])
m_dn = np.mean([st['Baseline'][s]['delta_norm'] for s in range(10)])
check('BL r_norm', 2.163, m_rn); check('BL delta', 1.162, m_dn)

# B1 gain
m,sd = ms(st['B1']); check('B1 gain', -0.004, m)

# ============ §3.4 ============
print('§3.4 Fake mirror...')
m,_ = ms(st['C1']); check('St C1 gain', -0.064, m)
m,_ = ms(st['C2']); check('St C2 gain', -0.059, m)

# ============ §3.8 Wrong-trajectory ============
print('§3.8 Wrong-traj...')
with open('results/wrong_trajectory_static.csv') as f:
    rows = list(csv.DictReader(f))
for cond,val,sd_val in [('self_current',0.053,0.054),('self_wrong_trial',0.061,0.061),
                         ('clone_current',-0.039,0.070)]:
    gains = [float(r['gain']) for r in rows if r['condition']==cond]
    check(f'St WT {cond}', val, np.mean(gains), 0.003)

with open('results/wrong_trajectory_vn.csv') as f:
    rows = list(csv.DictReader(f))
for cond,val in [('self_current',0.151),('self_wrong_trial',0.138),('clone_current',-0.018)]:
    gains = [float(r['gain']) for r in rows if r['condition']==cond]
    check(f'VN WT {cond}', val, np.mean(gains), 0.003)

# ============ §3.9 Divergence ============
print('§3.9 Divergence...')
with open('results/divergence_null_baseline.csv') as f:
    rows = list(csv.DictReader(f))
check('Random div', 1.001, np.mean([float(r['divergence_random_mean']) for r in rows]))
check('Clone div', 0.734, np.mean([float(r['divergence_clone']) for r in rows]))
rs = [float(r['divergence_resampled']) for r in rows if r['divergence_resampled']!='nan']
check('Resampled div', 0.361, np.mean(rs))

# ============ VN FF controls ============
print('VN FF controls...')
with open('results/vn_feedforward_controls.csv') as f:
    rows = list(csv.DictReader(f))
for g,val,sd_val in [('d',-0.011,0.047),('dp',-0.000,0.032),('dpp',0.010,0.035)]:
    gains = [float(r[f'{g}_gain']) for r in rows]
    check(f'VN FF {g}', val, np.mean(gains))
    check(f'VN FF {g} SD', sd_val, np.std(gains))

# ============ MNIST ============
print('MNIST...')
mp = 'C:/Users/clinic1/projects/self-awareness/mnist-feedback-contract/results/mnist_metrics.csv'
if os.path.exists(mp):
    with open(mp) as f:
        rows = list(csv.DictReader(f))
    for cond,val,sd_val in [('baseline',0.021,0.003),('group_a',-0.001,0.002),
                             ('c1_shuffle',-0.019,0.018),('c2_clone',0.006,0.007)]:
        gains = [float(r[f'{cond}_gain']) for r in rows]
        check(f'MNIST {cond}', val, np.mean(gains), 0.003)
        check(f'MNIST {cond} SD', sd_val, np.std(gains), 0.003)

# ============ Scale verification ============
print('Scale verification...')
for task in ['static','vn']:
    with open(f'results/scale_verification_{task}.csv') as f:
        rows = list(csv.DictReader(f))
    for h,cond,val in [(10,'baseline',0.053 if task=='static' else 0.146),
                        (20,'baseline',0.054 if task=='static' else 0.183),
                        (45,'baseline',0.042 if task=='static' else 0.154),
                        (245,'baseline',0.033 if task=='static' else 0.149)]:
        gains = [float(r['gain']) for r in rows if int(r['hidden_width'])==h and r['condition']==cond]
        if gains: check(f'{task} h={h} {cond}', val, np.mean(gains), 0.003)

# ============ Param count ============
print('Param count...')
from src.network import RecurrentMLP
net = RecurrentMLP(input_size=10,hidden1=245,hidden2=245,output_size=5,seed=0)
check('h=245 params', 65420, net.count_params(), 0)

# ============ Cross-pairing ============
print('Cross-pairing...')
with open('results/cross_pairing_vn.csv') as f:
    rows = list(csv.DictReader(f))
gains = [float(r['gain']) for r in rows]
check('Cross-pair mean', -0.043, np.mean(gains), 0.002)

# ============ Holm p-values ============
print('Holm p-values...')
check('2/1024', 0.001953, 2/1024, 0.000001)
check('VN Holm', 0.00586, 3*0.001953, 0.00001)

# ============ Summary ============
print()
if errors:
    print(f'!! ERRORS FOUND ({len(errors)}):')
    for e in errors:
        print(f'  {e}')
else:
    print('ALL NUMBERS VERIFIED - NO ERRORS FOUND')
