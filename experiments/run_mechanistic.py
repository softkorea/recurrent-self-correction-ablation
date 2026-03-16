"""WS1: Mechanistic Analysis — Activation Logging + Trajectory + W_rec Dissection.

Produces:
1. Activation traces for all 10 models × 200 test samples × 3 timesteps
2. Trial classification (corrected/stable_correct/stable_incorrect/over_corrected)
3. PCA trajectory visualization (Hero Image)
4. W_rec heatmap and per-trial decomposition
5. Clone feedback geometric analysis (cosine divergence)

Multiprocessing for parallel model evaluation.

Usage:
    python experiments/run_mechanistic.py
"""

import sys
import os
import time
import numpy as np
from multiprocessing import Pool, cpu_count

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.network import RecurrentMLP
from src.training import generate_data, train, softmax
from src.ablation import forward_sequence_with_clone


def train_model(seed, noise_level=0.5, n_samples=200, epochs=500,
                lr=0.01, tau=2.0):
    net = RecurrentMLP(input_size=10, hidden1=10, hidden2=10,
                       output_size=5, seed=seed, feedback_tau=tau)
    X, y = generate_data(n_samples, noise_level=noise_level, seed=seed)
    train(net, X, y, epochs=epochs, lr=lr,
          time_weights=[0.0, 0.2, 1.0])
    return net


def collect_activations(net, X, y, label='baseline'):
    """Collect full activation traces for all samples.

    Returns dict with arrays indexed by (sample, timestep).
    """
    n = len(X)
    T = 3

    traces = {
        'label': label,
        'raw_feedback': np.zeros((n, T, 5)),
        'scaled_feedback': np.zeros((n, T, 5)),
        'feedback_contrib': np.zeros((n, T, 10)),
        'ff_contrib': np.zeros((n, T, 10)),
        'z_h1': np.zeros((n, T, 10)),
        'a_h1': np.zeros((n, T, 10)),
        'z_h2': np.zeros((n, T, 10)),
        'a_h2': np.zeros((n, T, 10)),
        'output': np.zeros((n, T, 5)),
        'softmax': np.zeros((n, T, 5)),
        'pred': np.zeros((n, T), dtype=int),
        'true': np.zeros(n, dtype=int),
        'trial_type': [],  # corrected, stable_correct, stable_incorrect, over_corrected
    }

    for i in range(n):
        outputs, caches = net.forward_sequence(X[i], T=T)
        true_cls = np.argmax(y[i])
        traces['true'][i] = true_cls

        for t in range(T):
            c = caches[t]
            traces['scaled_feedback'][i, t] = c['feedback']
            traces['feedback_contrib'][i, t] = c['rec_contrib']
            traces['z_h1'][i, t] = c['z_h1']
            traces['a_h1'][i, t] = c['a_h1']
            traces['z_h2'][i, t] = c['z_h2']
            traces['a_h2'][i, t] = c['a_h2']
            traces['output'][i, t] = c['output']
            traces['softmax'][i, t] = softmax(c['output'])
            traces['pred'][i, t] = np.argmax(c['output'])

            if t > 0:
                traces['raw_feedback'][i, t] = outputs[t - 1]
            # ff contribution: x @ W_ih1 + b_h1
            traces['ff_contrib'][i, t] = X[i] @ net.W_ih1 + net.b_h1

        # Trial classification
        correct_t1 = (traces['pred'][i, 0] == true_cls)
        correct_t3 = (traces['pred'][i, 2] == true_cls)
        if not correct_t1 and correct_t3:
            traces['trial_type'].append('corrected')
        elif correct_t1 and correct_t3:
            traces['trial_type'].append('stable_correct')
        elif not correct_t1 and not correct_t3:
            traces['trial_type'].append('stable_incorrect')
        else:
            traces['trial_type'].append('over_corrected')

    traces['trial_type'] = np.array(traces['trial_type'])
    return traces


def collect_clone_activations(target_net, clone_net, X, y):
    """Collect activation traces with clone feedback (C2)."""
    n = len(X)
    T = 3

    traces = {
        'label': 'clone',
        'a_h1': np.zeros((n, T, 10)),
        'output': np.zeros((n, T, 5)),
        'pred': np.zeros((n, T), dtype=int),
        'true': np.zeros(n, dtype=int),
        'trial_type': [],
        # Clone-specific: self vs clone feedback comparison
        'self_feedback_contrib': np.zeros((n, T, 10)),
        'clone_feedback_contrib': np.zeros((n, T, 10)),
        'cosine_divergence': np.zeros((n, T)),
    }

    for i in range(n):
        # Clone forward
        outputs_c2, caches_c2 = forward_sequence_with_clone(
            target_net, clone_net, X[i], T=T)
        # Self forward (for reference)
        outputs_self, caches_self = target_net.forward_sequence(X[i], T=T)

        true_cls = np.argmax(y[i])
        traces['true'][i] = true_cls

        for t in range(T):
            traces['a_h1'][i, t] = caches_c2[t]['a_h1']
            traces['output'][i, t] = caches_c2[t]['output']
            traces['pred'][i, t] = np.argmax(caches_c2[t]['output'])

            if t > 0:
                # Self feedback contribution
                self_fb = np.tanh(outputs_self[t - 1] / target_net.feedback_tau)
                self_contrib = self_fb @ target_net.W_rec
                traces['self_feedback_contrib'][i, t] = self_contrib

                # Clone feedback contribution
                clone_fb = caches_c2[t]['feedback']
                clone_contrib = clone_fb @ target_net.W_rec
                traces['clone_feedback_contrib'][i, t] = clone_contrib

                # Cosine divergence
                norm_s = np.linalg.norm(self_contrib)
                norm_c = np.linalg.norm(clone_contrib)
                if norm_s > 1e-10 and norm_c > 1e-10:
                    cos_sim = np.dot(self_contrib, clone_contrib) / (norm_s * norm_c)
                    traces['cosine_divergence'][i, t] = 1 - cos_sim
                else:
                    traces['cosine_divergence'][i, t] = 1.0

        correct_t1 = (traces['pred'][i, 0] == true_cls)
        correct_t3 = (traces['pred'][i, 2] == true_cls)
        if not correct_t1 and correct_t3:
            traces['trial_type'].append('corrected')
        elif correct_t1 and correct_t3:
            traces['trial_type'].append('stable_correct')
        elif not correct_t1 and not correct_t3:
            traces['trial_type'].append('stable_incorrect')
        else:
            traces['trial_type'].append('over_corrected')

    traces['trial_type'] = np.array(traces['trial_type'])
    return traces


def plot_trajectory_hero(all_baseline_traces, all_clone_traces, save_dir):
    """Hero Image: PCA trajectory plot.

    Baseline corrected trials converging to class centroids,
    vs C2 trajectories diverging toward wrong attractors.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    # Aggregate H1 activations from all 10 models
    all_a_h1_t1 = []  # for PCA fitting
    baseline_trajs = {'corrected': [], 'stable_correct': [],
                      'stable_incorrect': [], 'over_corrected': []}
    clone_trajs = {'corrected': [], 'stable_correct': [],
                   'stable_incorrect': [], 'over_corrected': []}

    for traces in all_baseline_traces:
        n = len(traces['trial_type'])
        all_a_h1_t1.append(traces['a_h1'][:, 0, :])
        for i in range(n):
            ttype = traces['trial_type'][i]
            traj = traces['a_h1'][i]  # (3, 10)
            baseline_trajs[ttype].append(traj)

    for traces in all_clone_traces:
        n = len(traces['trial_type'])
        for i in range(n):
            ttype = traces['trial_type'][i]
            traj = traces['a_h1'][i]  # (3, 10)
            clone_trajs[ttype].append(traj)

    # PCA on Baseline t=1 activations
    all_a_h1_t1 = np.vstack(all_a_h1_t1)
    pca = PCA(n_components=2)
    pca.fit(all_a_h1_t1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel 1: Baseline trajectories
    ax = axes[0]
    ax.set_title('Baseline (self-feedback)', fontsize=13)
    colors = {'corrected': '#2ecc71', 'stable_correct': '#3498db',
              'stable_incorrect': '#e74c3c', 'over_corrected': '#f39c12'}
    labels = {'corrected': 'Corrected', 'stable_correct': 'Stable correct',
              'stable_incorrect': 'Stable incorrect', 'over_corrected': 'Over-corrected'}

    for ttype in ['stable_correct', 'corrected', 'stable_incorrect', 'over_corrected']:
        trajs = baseline_trajs[ttype]
        if not trajs:
            continue
        # Sample up to 50 for visibility
        n_show = min(50, len(trajs))
        for traj in trajs[:n_show]:
            proj = pca.transform(traj)  # (3, 2)
            ax.plot(proj[:, 0], proj[:, 1], '-', color=colors[ttype],
                    alpha=0.15, linewidth=0.8)
            ax.plot(proj[2, 0], proj[2, 1], 'o', color=colors[ttype],
                    alpha=0.3, markersize=3)
        # Centroid trajectory
        if trajs:
            mean_traj = np.mean(trajs, axis=0)
            proj_mean = pca.transform(mean_traj)
            ax.plot(proj_mean[:, 0], proj_mean[:, 1], '-',
                    color=colors[ttype], linewidth=2.5, label=f'{labels[ttype]} ({len(trajs)})')
            ax.plot(proj_mean[0, 0], proj_mean[0, 1], 's',
                    color=colors[ttype], markersize=8)
            ax.plot(proj_mean[2, 0], proj_mean[2, 1], '*',
                    color=colors[ttype], markersize=12)

    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3)

    # Panel 2: C2 (clone feedback) trajectories
    ax = axes[1]
    ax.set_title('C2 (clone feedback)', fontsize=13)

    for ttype in ['stable_correct', 'corrected', 'stable_incorrect', 'over_corrected']:
        trajs = clone_trajs[ttype]
        if not trajs:
            continue
        n_show = min(50, len(trajs))
        for traj in trajs[:n_show]:
            proj = pca.transform(traj)
            ax.plot(proj[:, 0], proj[:, 1], '-', color=colors[ttype],
                    alpha=0.15, linewidth=0.8)
            ax.plot(proj[2, 0], proj[2, 1], 'o', color=colors[ttype],
                    alpha=0.3, markersize=3)
        if trajs:
            mean_traj = np.mean(trajs, axis=0)
            proj_mean = pca.transform(mean_traj)
            ax.plot(proj_mean[:, 0], proj_mean[:, 1], '-',
                    color=colors[ttype], linewidth=2.5, label=f'{labels[ttype]} ({len(trajs)})')
            ax.plot(proj_mean[0, 0], proj_mean[0, 1], 's',
                    color=colors[ttype], markersize=8)
            ax.plot(proj_mean[2, 0], proj_mean[2, 1], '*',
                    color=colors[ttype], markersize=12)

    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3)

    plt.suptitle('Hidden Layer 1 Trajectories (t=1→t=3) in PCA Space\n'
                 '■ = t=1 start, ★ = t=3 end', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'trajectory_hero.png'), dpi=200,
                bbox_inches='tight')
    plt.close()
    print("  → trajectory_hero.png")


def plot_wrec_heatmap(models, save_dir):
    """W_rec heatmap: 5×10 matrix for each model + aggregate."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    vmin = min(m.W_rec.min() for m in models.values())
    vmax = max(m.W_rec.max() for m in models.values())
    vabs = max(abs(vmin), abs(vmax))

    for idx, (seed, net) in enumerate(sorted(models.items())):
        ax = axes[idx // 5, idx % 5]
        im = ax.imshow(net.W_rec, cmap='RdBu_r', vmin=-vabs, vmax=vabs,
                       aspect='auto')
        ax.set_title(f'Seed {seed}', fontsize=10)
        ax.set_xlabel('H1 neuron')
        ax.set_ylabel('Output dim')
        ax.set_xticks(range(10))
        ax.set_yticks(range(5))

    plt.suptitle('W_rec (Output→H1) Across 10 Models', fontsize=14)
    plt.tight_layout()
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6, label='Weight value')
    plt.savefig(os.path.join(save_dir, 'wrec_heatmap.png'), dpi=200,
                bbox_inches='tight')
    plt.close()
    print("  → wrec_heatmap.png")


def plot_cosine_divergence(all_clone_traces, save_dir):
    """Clone feedback cosine divergence: self vs clone W_rec contribution."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # Aggregate cosine divergence at t=2 (most informative timestep)
    divergences = []
    degradations = []  # 1 if over-corrected or stable_incorrect under C2

    for traces in all_clone_traces:
        n = len(traces['trial_type'])
        for i in range(n):
            div = traces['cosine_divergence'][i, 1]  # t=2
            divergences.append(div)
            degraded = traces['trial_type'][i] in ('stable_incorrect', 'over_corrected')
            degradations.append(int(degraded))

    divergences = np.array(divergences)
    degradations = np.array(degradations)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel 1: Distribution by trial outcome
    ax = axes[0]
    for ttype, color, label in [
        ('stable_correct', '#3498db', 'Stable correct'),
        ('corrected', '#2ecc71', 'Corrected'),
        ('stable_incorrect', '#e74c3c', 'Stable incorrect'),
        ('over_corrected', '#f39c12', 'Over-corrected'),
    ]:
        vals = []
        for traces in all_clone_traces:
            for i in range(len(traces['trial_type'])):
                if traces['trial_type'][i] == ttype:
                    vals.append(traces['cosine_divergence'][i, 1])
        if vals:
            ax.hist(vals, bins=20, alpha=0.5, color=color, label=f'{label} ({len(vals)})')

    ax.set_xlabel('Cosine Divergence (self vs clone feedback contrib)')
    ax.set_ylabel('Count')
    ax.set_title('Divergence by C2 Trial Outcome')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 2: Binned degradation rate vs divergence
    ax = axes[1]
    n_bins = 10
    bin_edges = np.linspace(0, divergences.max() + 1e-6, n_bins + 1)
    bin_centers = []
    bin_rates = []
    for b in range(n_bins):
        mask = (divergences >= bin_edges[b]) & (divergences < bin_edges[b + 1])
        if mask.sum() > 5:  # minimum count
            bin_centers.append((bin_edges[b] + bin_edges[b + 1]) / 2)
            bin_rates.append(degradations[mask].mean())

    ax.plot(bin_centers, bin_rates, 'o-', color='#e74c3c', linewidth=2)
    ax.set_xlabel('Cosine Divergence')
    ax.set_ylabel('Degradation Rate')
    ax.set_title('Degradation Rate vs Feedback Divergence')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    plt.suptitle('Clone Feedback Geometric Analysis', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'cosine_divergence.png'), dpi=200,
                bbox_inches='tight')
    plt.close()
    print("  → cosine_divergence.png")


def plot_interpolation_phase(save_dir):
    """Phase diagram from WS2 interpolation results."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    csv_path = os.path.join(save_dir, 'interpolation.csv')
    if not os.path.exists(csv_path):
        print("  (interpolation.csv not found, skipping phase diagram)")
        return

    import csv
    rows = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    alphas = sorted(set(float(r['alpha']) for r in rows))
    types = ['zero', 'shuffle', 'clone']
    colors = {'zero': '#3498db', 'shuffle': '#e74c3c', 'clone': '#9b59b6'}
    labels = {'zero': 'Self–Zero', 'shuffle': 'Self–Shuffle', 'clone': 'Self–Clone'}

    fig, ax = plt.subplots(figsize=(8, 5))

    for itype in types:
        means = []
        stds = []
        for alpha in alphas:
            gains = [float(r['gain']) for r in rows
                     if float(r['alpha']) == alpha and r['interp_type'] == itype]
            means.append(np.mean(gains))
            stds.append(np.std(gains))

        means = np.array(means)
        stds = np.array(stds)
        ax.plot(alphas, means, 'o-', color=colors[itype], linewidth=2,
                label=labels[itype], markersize=6)
        ax.fill_between(alphas, means - stds, means + stds,
                        color=colors[itype], alpha=0.15)

    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('α (self-feedback fraction)', fontsize=12)
    ax.set_ylabel('Correction Gain', fontsize=12)
    ax.set_title('Feedback Interpolation Phase Diagram\n'
                 'feedback = α·y_self + (1-α)·y_other', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.02, 1.02)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'interpolation_phase.png'), dpi=200,
                bbox_inches='tight')
    plt.close()
    print("  → interpolation_phase.png")


def main():
    seeds = list(range(10))
    donor_seeds = list(range(100, 110))
    noise_level = 0.5
    n_test = 200

    results_dir = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), 'results')

    total_start = time.time()
    print("=" * 60)
    print("WS1: Mechanistic Analysis")
    print("=" * 60)

    # Phase 1: Train models
    print("\n[Phase 1] Training models...")
    target_models = {}
    for s in seeds:
        t0 = time.time()
        target_models[s] = train_model(s)
        print(f"  target seed={s}: {time.time()-t0:.1f}s")

    clone_models = {}
    for ds in donor_seeds:
        t0 = time.time()
        clone_models[ds] = train_model(ds)
        print(f"  donor  seed={ds}: {time.time()-t0:.1f}s")

    # Phase 2: Collect activations
    print("\n[Phase 2] Collecting activation traces...")
    all_baseline_traces = []
    all_clone_traces = []

    for s_idx, s in enumerate(seeds):
        X_test, y_test = generate_data(n_test, noise_level=noise_level,
                                       seed=1000 + s)
        net = target_models[s]
        clone = clone_models[donor_seeds[s_idx]]

        t0 = time.time()
        baseline_traces = collect_activations(net, X_test, y_test, f'baseline_s{s}')
        clone_traces = collect_clone_activations(net, clone, X_test, y_test)

        all_baseline_traces.append(baseline_traces)
        all_clone_traces.append(clone_traces)

        # Summary
        tt = baseline_traces['trial_type']
        n_corr = np.sum(tt == 'corrected')
        n_sc = np.sum(tt == 'stable_correct')
        n_si = np.sum(tt == 'stable_incorrect')
        n_oc = np.sum(tt == 'over_corrected')
        print(f"  seed={s}: corrected={n_corr}, stable_correct={n_sc}, "
              f"stable_incorrect={n_si}, over_corrected={n_oc}  "
              f"({time.time()-t0:.1f}s)")

    # Phase 3: Generate visualizations
    print("\n[Phase 3] Generating visualizations...")

    try:
        plot_trajectory_hero(all_baseline_traces, all_clone_traces, results_dir)
    except ImportError as e:
        print(f"  (sklearn not available for PCA, using manual PCA: {e})")
        # Fallback: manual PCA
        plot_trajectory_hero_manual(all_baseline_traces, all_clone_traces, results_dir)

    plot_wrec_heatmap(target_models, results_dir)
    plot_cosine_divergence(all_clone_traces, results_dir)
    plot_interpolation_phase(results_dir)

    # Save numeric summaries
    print("\n[Phase 4] Numeric summaries...")

    # Cosine divergence stats
    all_div_t2 = []
    for traces in all_clone_traces:
        all_div_t2.extend(traces['cosine_divergence'][:, 1].tolist())
    all_div_t2 = np.array(all_div_t2)
    print(f"  Cosine divergence (t=2): mean={all_div_t2.mean():.4f}, "
          f"std={all_div_t2.std():.4f}, "
          f"median={np.median(all_div_t2):.4f}")

    # Trial type distribution aggregate
    total_counts = {'corrected': 0, 'stable_correct': 0,
                    'stable_incorrect': 0, 'over_corrected': 0}
    for traces in all_baseline_traces:
        for tt in traces['trial_type']:
            total_counts[tt] += 1
    total = sum(total_counts.values())
    print(f"  Baseline trial distribution (N={total}):")
    for k, v in total_counts.items():
        print(f"    {k}: {v} ({v/total:.1%})")

    # W_rec structural analysis
    print(f"\n  W_rec structural consistency across models:")
    wrecs = np.stack([target_models[s].W_rec for s in seeds])  # (10, 5, 10)
    sign_consistency = np.mean(np.sign(wrecs) == np.sign(wrecs.mean(axis=0)), axis=0)
    print(f"    Mean sign consistency: {sign_consistency.mean():.3f}")
    print(f"    Min sign consistency:  {sign_consistency.min():.3f}")

    total_time = time.time() - total_start
    print(f"\nTotal time: {total_time:.0f}s")


def plot_trajectory_hero_manual(all_baseline_traces, all_clone_traces, save_dir):
    """Manual PCA fallback (no sklearn dependency)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # Aggregate t=1 activations for PCA
    all_a_h1_t1 = np.vstack([t['a_h1'][:, 0, :] for t in all_baseline_traces])

    # Manual PCA via SVD
    mean = all_a_h1_t1.mean(axis=0)
    centered = all_a_h1_t1 - mean
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    components = Vt[:2]  # top 2 PCs
    explained_var = (S[:2] ** 2) / (S ** 2).sum()

    def project(x):
        return (x - mean) @ components.T

    colors = {'corrected': '#2ecc71', 'stable_correct': '#3498db',
              'stable_incorrect': '#e74c3c', 'over_corrected': '#f39c12'}
    labels = {'corrected': 'Corrected', 'stable_correct': 'Stable correct',
              'stable_incorrect': 'Stable incorrect', 'over_corrected': 'Over-corrected'}

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for panel_idx, (all_traces, title) in enumerate([
        (all_baseline_traces, 'Baseline (self-feedback)'),
        (all_clone_traces, 'C2 (clone feedback)')
    ]):
        ax = axes[panel_idx]
        ax.set_title(title, fontsize=13)

        trajs_by_type = {}
        for traces in all_traces:
            n = len(traces['trial_type'])
            for i in range(n):
                ttype = traces['trial_type'][i]
                if ttype not in trajs_by_type:
                    trajs_by_type[ttype] = []
                trajs_by_type[ttype].append(traces['a_h1'][i])

        for ttype in ['stable_correct', 'corrected', 'stable_incorrect', 'over_corrected']:
            trajs = trajs_by_type.get(ttype, [])
            if not trajs:
                continue
            n_show = min(50, len(trajs))
            for traj in trajs[:n_show]:
                proj = project(traj)
                ax.plot(proj[:, 0], proj[:, 1], '-', color=colors[ttype],
                        alpha=0.15, linewidth=0.8)
            if trajs:
                mean_traj = np.mean(trajs, axis=0)
                proj_mean = project(mean_traj)
                ax.plot(proj_mean[:, 0], proj_mean[:, 1], '-',
                        color=colors[ttype], linewidth=2.5,
                        label=f'{labels[ttype]} ({len(trajs)})')
                ax.plot(proj_mean[0, 0], proj_mean[0, 1], 's',
                        color=colors[ttype], markersize=8)
                ax.plot(proj_mean[2, 0], proj_mean[2, 1], '*',
                        color=colors[ttype], markersize=12)

        ax.set_xlabel(f'PC1 ({explained_var[0]:.1%})')
        ax.set_ylabel(f'PC2 ({explained_var[1]:.1%})')
        ax.legend(fontsize=9, loc='best')
        ax.grid(True, alpha=0.3)

    plt.suptitle('Hidden Layer 1 Trajectories (t=1→t=3) in PCA Space\n'
                 '■ = t=1 start, ★ = t=3 end', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'trajectory_hero.png'), dpi=200,
                bbox_inches='tight')
    plt.close()
    print("  → trajectory_hero.png")


if __name__ == '__main__':
    main()
