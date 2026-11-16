import csv
import math

records = list(csv.DictReader(open('results/experiment_records.csv', encoding='utf-8')))

clusters = sorted(list(set(r['problem_id'] for r in records)))
G = len(clusters)
N = len(records)
K = 4

beta = [0.15142, -1.43007, -2.02323, 1.50493]

def get_x(r):
    exp = 1.0 if not r['is_spec_only'] == 'True' else 0.0
    div = 1.0 if r['is_cross_family'] == 'True' else 0.0
    return [1.0, exp, div, exp * div]

X = [get_x(r) for r in records]
Y = [1.0 if r['detects_defect'] == 'True' else 0.0 for r in records]
P = []
W = []
for x in X:
    xb = sum(xi * bi for xi, bi in zip(x, beta))
    p = 1.0 / (1.0 + math.exp(-xb))
    P.append(p)
    W.append(p * (1.0 - p))

H = [[0.0]*4 for _ in range(4)]
for i in range(len(records)):
    w_i = W[i]
    x_i = X[i]
    for j in range(4):
        for k in range(4):
            H[j][k] += w_i * x_i[j] * x_i[k]

def invert_4x4(A):
    M = [A[i][:] + [1.0 if i == j else 0.0 for j in range(4)] for i in range(4)]
    for i in range(4):
        pivot = M[i][i]
        if abs(pivot) < 1e-12:
            raise ValueError('Singular matrix')
        for j in range(8):
            M[i][j] /= pivot
        for r in range(4):
            if r != i:
                factor = M[r][i]
                for j in range(8):
                    M[r][j] -= factor * M[i][j]
    return [[M[i][j+4] for j in range(4)] for i in range(4)]

H_inv = invert_4x4(H)
se_unclustered = [math.sqrt(H_inv[i][i]) for i in range(4)]

B = [[0.0]*4 for _ in range(4)]
dfc = (G / (G - 1.0)) * ((N - 1.0) / (N - K))

for g in clusters:
    s_g = [0.0]*4
    for i, r in enumerate(records):
        if r['problem_id'] == g:
            e_i = Y[i] - P[i]
            for j in range(4):
                s_g[j] += X[i][j] * e_i
    for j in range(4):
        for k in range(4):
            B[j][k] += s_g[j] * s_g[k]

for j in range(4):
    for k in range(4):
        B[j][k] *= dfc

Temp = [[0.0]*4 for _ in range(4)]
for i in range(4):
    for j in range(4):
        Temp[i][j] = sum(B[i][k] * H_inv[k][j] for k in range(4))

V_cluster = [[0.0]*4 for _ in range(4)]
for i in range(4):
    for j in range(4):
        V_cluster[i][j] = sum(H_inv[i][k] * Temp[k][j] for k in range(4))

se_clustered = [math.sqrt(V_cluster[i][i]) for i in range(4)]
t_clustered = [beta[i] / se_clustered[i] for i in range(4)]

# Student-t integration for df=14
# c = Gamma(7.5)/(sqrt(14*pi)*Gamma(7)) = 0.373656
c = 0.373656
def t_tail(t_val, df=14):
    limit = abs(t_val)
    steps = 50000
    b = 50.0
    dx = (b - limit) / steps
    area = 0.0
    for i in range(steps):
        x = limit + (i + 0.5) * dx
        area += (1.0 + x*x/df)**(-0.5*(df+1)) * dx
    return 2.0 * c * area

# Critical t for 95% CI with df=14: t_crit = 2.1448
t_crit = 2.1448

params = ['Intercept', 'Code Exposure', 'Model Diversity', 'Exposure x Diversity']
print('\n=== Corrected Parameter Estimates (t-distribution with df=14) ===')
for i in range(4):
    p_t = t_tail(t_clustered[i], 14)
    # Robust 95% CI for beta
    b_low = beta[i] - t_crit * se_clustered[i]
    b_high = beta[i] + t_crit * se_clustered[i]
    or_val = math.exp(beta[i])
    or_low = math.exp(b_low)
    or_high = math.exp(b_high)
    print(f'{params[i]}:')
    print(f'  Beta = {beta[i]:+.4f}, Robust SE = {se_clustered[i]:.4f}, t(14) = {t_clustered[i]:+.3f}, p = {p_t:.6f}')
    print(f'  OR = {or_val:.4f}, Robust 95% CI (OR) = [{or_low:.4f}, {or_high:.4f}]')

from collections import defaultdict
by_task = defaultdict(lambda: defaultdict(list))
for r in records:
    by_task[r['problem_id']][r['condition']].append(r['detects_defect'] == 'True')

diffs = []
for t in clusters:
    t0_rate = sum(by_task[t]['T0_Spec_Same']) / len(by_task[t]['T0_Spec_Same'])
    t1_rate = sum(by_task[t]['T1_Code_Same']) / len(by_task[t]['T1_Code_Same'])
    diffs.append((t0_rate - t1_rate) * 100.0)

mean_d = sum(diffs) / len(diffs)
var_d = sum((x - mean_d)**2 for x in diffs) / (len(diffs) - 1)
se_d = math.sqrt(var_d) / math.sqrt(len(diffs))
t_task = mean_d / se_d
p_task = t_tail(t_task, df=14)

print('\n=== Task-Level Sensitivity Analysis (N=15 Clusters) ===')
print(f'Mean Difference = {mean_d:.2f} pp, Sample SD = {math.sqrt(var_d):.4f}, SE = {se_d:.4f}')
print(f'Paired t(14) = {t_task:.3f}, two-sided p = {p_task:.6f}')
print(f'Wilcoxon signed-rank test on 14 non-zero tasks: W = 0, two-sided p = {2.0 * (0.5**14):.6f} (one-sided p = {0.5**14:.6f})')

