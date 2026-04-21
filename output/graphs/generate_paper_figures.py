#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path('/home/mila/e/echchabo/projects/llm-course-project')
OUT = ROOT / 'output' / 'graphs'
ROUTER_DIR = ROOT / 'output' / 'retrieval' / 'method_router'

QUERY_MAP_CANDIDATES = [
    Path('/network/scratch/y/yuyan.chen/inquire/inquire/inquire_queries_test.csv'),
    ROOT / 'inquire_queries_test.csv',
]

T2I_FILES = [
    ROOT / 'results_rerank_with_clip_test_bio.csv',
    ROOT / 'results_rerank_with_clip_test_siglip.csv',
]
I2I_FILE = ROOT / 'archive' / 'results_top8_filtered_topk_all_models_test.csv'
T2T_PROXY_FILE = ROOT / 'results_rerank_with_llm_test_qwen4b_cap.csv'
CAPTION_CACHE = ROUTER_DIR / 'caption_cache_test.json'

CATEGORY_ORDER = ['Appearance', 'Behavior', 'Context', 'Species']
MODEL_ORDER = ['vit-b-32', 'bioclip', 'biocap', 'siglip-vit-b-16']

# Alias siglip naming across result files.
MODEL_ALIASES = {
    'siglip-so400m-14-384': 'siglip-vit-b-16',
}


def normalize_model(name: str) -> str:
    return MODEL_ALIASES.get(name, name)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open(newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def load_query_map() -> dict[str, str]:
    for p in QUERY_MAP_CANDIDATES:
        if not p.exists():
            continue
        mapping: dict[str, str] = {}
        for row in read_csv_rows(p):
            q = (row.get('query_text') or row.get('query') or '').strip()
            cat = (row.get('supercategory') or row.get('category') or '').strip()
            if q:
                mapping[q] = cat
        if mapping:
            return mapping
    raise FileNotFoundError('Could not find query category mapping CSV.')


def load_fixed_method_scores() -> tuple[dict[tuple[str, str], float], dict[tuple[str, str], float], dict[str, float]]:
    t2i: dict[tuple[str, str], float] = {}
    for f in T2I_FILES:
        if not f.exists():
            continue
        for row in read_csv_rows(f):
            model = normalize_model((row.get('model') or '').strip())
            query = (row.get('query') or row.get('query_text') or '').strip()
            ap = row.get('ap')
            if not model or not query or ap is None:
                continue
            try:
                t2i[(model, query)] = float(ap)
            except ValueError:
                continue

    i2i: dict[tuple[str, str], float] = {}
    if I2I_FILE.exists():
        for row in read_csv_rows(I2I_FILE):
            model = normalize_model((row.get('model') or '').strip())
            query = (row.get('query') or row.get('query_text') or '').strip()
            ap = row.get('ap')
            if not model or not query or ap is None:
                continue
            try:
                i2i[(model, query)] = float(ap)
            except ValueError:
                continue

    t2t_proxy: dict[str, float] = {}
    if T2T_PROXY_FILE.exists():
        for row in read_csv_rows(T2T_PROXY_FILE):
            query = (row.get('query') or row.get('query_text') or '').strip()
            ap = row.get('ap')
            if not query or ap is None:
                continue
            try:
                t2t_proxy[query] = float(ap)
            except ValueError:
                continue

    return t2i, i2i, t2t_proxy


def load_orchestrator_rows() -> dict[str, list[dict[str, str]]]:
    out: dict[str, list[dict[str, str]]] = {}
    for model in MODEL_ORDER:
        p = ROUTER_DIR / f'results_orchestrator_{model}_test.csv'
        if not p.exists():
            continue
        rows = read_csv_rows(p)
        out[model] = rows
    return out


def build_joined_records(
    query_map: dict[str, str],
    t2i_scores: dict[tuple[str, str], float],
    i2i_scores: dict[tuple[str, str], float],
    t2t_proxy: dict[str, float],
    orchestrator_rows: dict[str, list[dict[str, str]]],
) -> dict[str, list[dict[str, object]]]:
    joined: dict[str, list[dict[str, object]]] = {}
    for model, rows in orchestrator_rows.items():
        recs: list[dict[str, object]] = []
        for row in rows:
            q = (row.get('query') or '').strip()
            if not q:
                continue
            cat = query_map.get(q, 'Unknown')
            try:
                orch_ap = float(row.get('ap', 'nan'))
            except ValueError:
                orch_ap = float('nan')

            try:
                conf = float(row.get('router_confidence', 'nan'))
            except ValueError:
                conf = float('nan')

            recs.append(
                {
                    'model': model,
                    'query': q,
                    'category': cat,
                    'chosen_method': (row.get('chosen_method') or '').strip(),
                    'executed_method': (row.get('executed_method') or '').strip(),
                    'router_confidence': conf,
                    'router_reason': (row.get('router_reason') or '').strip(),
                    'num_web_images': int(float(row.get('num_web_images', '0') or 0)),
                    'num_web_candidates': int(float(row.get('num_web_candidates', '0') or 0)),
                    'ap_orch': orch_ap,
                    'ap_t2i': t2i_scores.get((model, q), float('nan')),
                    'ap_i2i': i2i_scores.get((model, q), float('nan')),
                    'ap_t2t_proxy': t2t_proxy.get(q, float('nan')),
                }
            )
        joined[model] = recs
    return joined


def savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches='tight')
    plt.close()


def fig1_routing_sankey(joined: dict[str, list[dict[str, object]]]) -> None:
    # Custom alluvial-like plot with weighted links.
    cats = CATEGORY_ORDER
    methods = ['i2i', 't2i', 't2t']

    counts_cm = Counter()
    counts_me = Counter()
    for recs in joined.values():
        for r in recs:
            counts_cm[(r['category'], r['chosen_method'])] += 1
            counts_me[(r['chosen_method'], r['executed_method'])] += 1

    cat_totals = {c: sum(counts_cm[(c, m)] for m in methods) for c in cats}
    ch_totals = {m: sum(counts_cm[(c, m)] for c in cats) for m in methods}
    ex_totals = {m: sum(counts_me[(c, m)] for c in methods) for m in methods}

    fig, ax = plt.subplots(figsize=(12, 6.5))
    x_cat, x_ch, x_ex = 0.1, 0.5, 0.9
    y0 = 0.05
    height = 0.9

    def node_spans(totals: dict[str, int], order: list[str]) -> dict[str, tuple[float, float]]:
        total = max(sum(totals.values()), 1)
        spans = {}
        y = y0
        for k in order:
            h = height * (totals.get(k, 0) / total)
            spans[k] = (y, y + h)
            y += h
        return spans

    cat_sp = node_spans(cat_totals, cats)
    ch_sp = node_spans(ch_totals, methods)
    ex_sp = node_spans(ex_totals, methods)

    colors = {'i2i': '#1f77b4', 't2i': '#ff7f0e', 't2t': '#2ca02c'}

    for c in cats:
        y1, y2 = cat_sp[c]
        ax.add_patch(plt.Rectangle((x_cat - 0.03, y1), 0.06, y2 - y1, color='#bbbbbb', alpha=0.8))
        ax.text(x_cat - 0.04, (y1 + y2) / 2, c, va='center', ha='right', fontsize=10)
    for m in methods:
        y1, y2 = ch_sp[m]
        ax.add_patch(plt.Rectangle((x_ch - 0.03, y1), 0.06, y2 - y1, color=colors[m], alpha=0.5))
        ax.text(x_ch, y2 + 0.012, f'ch:{m}', va='bottom', ha='center', fontsize=9)
    for m in methods:
        y1, y2 = ex_sp[m]
        ax.add_patch(plt.Rectangle((x_ex - 0.03, y1), 0.06, y2 - y1, color=colors[m], alpha=0.8))
        ax.text(x_ex + 0.04, (y1 + y2) / 2, m, va='center', ha='left', fontsize=10)

    cat_offsets = {c: cat_sp[c][0] for c in cats}
    ch_in_offsets = {m: ch_sp[m][0] for m in methods}
    ch_out_offsets = {m: ch_sp[m][0] for m in methods}
    ex_offsets = {m: ex_sp[m][0] for m in methods}

    total_n = max(sum(cat_totals.values()), 1)
    scale = height / total_n

    for c in cats:
        for m in methods:
            v = counts_cm[(c, m)]
            if v == 0:
                continue
            y1a = cat_offsets[c]
            y1b = y1a + v * scale
            y2a = ch_in_offsets[m]
            y2b = y2a + v * scale
            ax.fill_between([x_cat + 0.03, x_ch - 0.03], [y1a, y2a], [y1b, y2b], color=colors[m], alpha=0.25)
            cat_offsets[c] += v * scale
            ch_in_offsets[m] += v * scale

    for cm in methods:
        for em in methods:
            v = counts_me[(cm, em)]
            if v == 0:
                continue
            y1a = ch_out_offsets[cm]
            y1b = y1a + v * scale
            y2a = ex_offsets[em]
            y2b = y2a + v * scale
            ax.fill_between([x_ch + 0.03, x_ex - 0.03], [y1a, y2a], [y1b, y2b], color=colors[cm], alpha=0.22)
            ch_out_offsets[cm] += v * scale
            ex_offsets[em] += v * scale

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('Routing Flow: Query Category -> Chosen Method -> Executed Method')
    for spine in ax.spines.values():
        spine.set_visible(False)
    savefig(OUT / 'fig01_routing_sankey.png')


def fig2_winrate_heatmap(joined: dict[str, list[dict[str, object]]]) -> None:
    # Win rate against best fixed method available (i2i, t2i, and t2t-proxy).
    arr = np.zeros((len(MODEL_ORDER), len(CATEGORY_ORDER)), dtype=float)

    for i, model in enumerate(MODEL_ORDER):
        by_cat = defaultdict(list)
        for r in joined.get(model, []):
            fixed = [r['ap_t2i'], r['ap_i2i'], r['ap_t2t_proxy']]
            fixed = [x for x in fixed if isinstance(x, float) and not math.isnan(x)]
            if not fixed or math.isnan(r['ap_orch']):
                continue
            win = 1.0 if r['ap_orch'] > max(fixed) else 0.0
            by_cat[r['category']].append(win)
        for j, cat in enumerate(CATEGORY_ORDER):
            vals = by_cat.get(cat, [])
            arr[i, j] = (100.0 * mean(vals)) if vals else np.nan

    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    im = ax.imshow(arr, cmap='YlGnBu', vmin=0, vmax=100)
    ax.set_xticks(range(len(CATEGORY_ORDER)), CATEGORY_ORDER)
    ax.set_yticks(range(len(MODEL_ORDER)), MODEL_ORDER)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            if np.isnan(arr[i, j]):
                txt = '-'
            else:
                txt = f'{arr[i, j]:.1f}'
            ax.text(j, i, txt, ha='center', va='center', fontsize=9, color='black')
    ax.set_title('Orchestrator Win Rate vs Best Fixed Method (%)')
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Win Rate (%)')
    savefig(OUT / 'fig02_orchestrator_winrate_heatmap.png')


def category_means(recs: list[dict[str, object]], key: str) -> dict[str, float]:
    g = defaultdict(list)
    for r in recs:
        v = r.get(key)
        if not isinstance(v, float) or math.isnan(v):
            continue
        g[r['category']].append(v)
    return {c: (mean(g[c]) if g[c] else float('nan')) for c in CATEGORY_ORDER}


def fig3_delta_ap_by_category(joined: dict[str, list[dict[str, object]]]) -> None:
    # Delta vs fixed methods (i2i, t2i, t2t-proxy).
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharey=True)
    axes = axes.flatten()

    x = np.arange(len(CATEGORY_ORDER))
    w = 0.24
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    for idx, model in enumerate(MODEL_ORDER):
        ax = axes[idx]
        recs = joined.get(model, [])
        orch = category_means(recs, 'ap_orch')
        t2i = category_means(recs, 'ap_t2i')
        i2i = category_means(recs, 'ap_i2i')
        t2t = category_means(recs, 'ap_t2t_proxy')

        d_i2i = [orch[c] - i2i[c] if not (math.isnan(orch[c]) or math.isnan(i2i[c])) else np.nan for c in CATEGORY_ORDER]
        d_t2i = [orch[c] - t2i[c] if not (math.isnan(orch[c]) or math.isnan(t2i[c])) else np.nan for c in CATEGORY_ORDER]
        d_t2t = [orch[c] - t2t[c] if not (math.isnan(orch[c]) or math.isnan(t2t[c])) else np.nan for c in CATEGORY_ORDER]

        ax.axhline(0, color='black', linewidth=0.9)
        ax.bar(x - w, d_i2i, width=w, label='- I2I', color=colors[0], alpha=0.85)
        ax.bar(x, d_t2i, width=w, label='- T2I', color=colors[1], alpha=0.85)
        ax.bar(x + w, d_t2t, width=w, label='- T2T proxy', color=colors[2], alpha=0.85)
        ax.set_xticks(x, CATEGORY_ORDER, rotation=20)
        ax.set_title(model)
        if idx % 2 == 0:
            ax.set_ylabel('Delta AP@50')
        if idx == 0:
            ax.legend(fontsize=8)

    fig.suptitle('Category-Wise Delta: Orchestrator - Fixed Methods')
    savefig(OUT / 'fig03_delta_ap_by_category.png')


def fig4_oracle_gap(joined: dict[str, list[dict[str, object]]]) -> None:
    labels = []
    router_vals = []
    oracle_vals = []

    for model in MODEL_ORDER:
        recs = joined.get(model, [])
        router = []
        oracle = []
        for r in recs:
            fixed = [r['ap_t2i'], r['ap_i2i'], r['ap_t2t_proxy']]
            fixed = [x for x in fixed if isinstance(x, float) and not math.isnan(x)]
            if not fixed or math.isnan(r['ap_orch']):
                continue
            router.append(r['ap_orch'])
            oracle.append(max(fixed))
        labels.append(model)
        router_vals.append(mean(router) if router else np.nan)
        oracle_vals.append(mean(oracle) if oracle else np.nan)

    x = np.arange(len(labels))
    w = 0.36
    plt.figure(figsize=(8.8, 4.3))
    plt.bar(x - w / 2, router_vals, width=w, label='Orchestrator AP@50', color='#4c78a8')
    plt.bar(x + w / 2, oracle_vals, width=w, label='Oracle AP@50', color='#f58518')
    for i, (r, o) in enumerate(zip(router_vals, oracle_vals)):
        if np.isnan(r) or np.isnan(o):
            continue
        plt.text(i, max(r, o) + 0.5, f'gap {o-r:.1f}', ha='center', fontsize=9)
    plt.xticks(x, labels)
    plt.ylabel('AP@50')
    plt.title('Orchestrator vs Oracle Upper Bound (using available fixed methods)')
    plt.legend()
    savefig(OUT / 'fig04_oracle_gap.png')


def write_oracle_results_csv(joined: dict[str, list[dict[str, object]]]) -> None:
    out_path = OUT / 'oracle_results.csv'
    rows: list[dict[str, object]] = []

    for model in MODEL_ORDER:
        recs = joined.get(model, [])
        router = []
        oracle = []
        for r in recs:
            fixed = [r['ap_t2i'], r['ap_i2i'], r['ap_t2t_proxy']]
            fixed = [x for x in fixed if isinstance(x, float) and not math.isnan(x)]
            if not fixed or math.isnan(r['ap_orch']):
                continue
            router.append(r['ap_orch'])
            oracle.append(max(fixed))

        router_mean = mean(router) if router else np.nan
        oracle_mean = mean(oracle) if oracle else np.nan
        gap = oracle_mean - router_mean if not (math.isnan(router_mean) or math.isnan(oracle_mean)) else np.nan
        rows.append(
            {
                'model': model,
                'n_queries_used': len(router),
                'router_ap50_mean': router_mean,
                'oracle_ap50_mean': oracle_mean,
                'oracle_minus_router_gap': gap,
            }
        )

    with out_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                'model',
                'n_queries_used',
                'router_ap50_mean',
                'oracle_ap50_mean',
                'oracle_minus_router_gap',
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def fig5_confidence_calibration(joined: dict[str, list[dict[str, object]]]) -> None:
    bins = np.linspace(0.0, 1.0, 11)
    centers = (bins[:-1] + bins[1:]) / 2.0

    plt.figure(figsize=(9.2, 4.8))
    for model in MODEL_ORDER:
        recs = joined.get(model, [])
        y = []
        for lo, hi in zip(bins[:-1], bins[1:]):
            bucket = [r for r in recs if isinstance(r['router_confidence'], float) and lo <= r['router_confidence'] < hi]
            if not bucket:
                y.append(np.nan)
                continue
            correct = [1.0 if r['chosen_method'] == r['executed_method'] and not r.get('fallback_reason') else 0.0 for r in bucket]
            y.append(100.0 * mean(correct))
        plt.plot(centers, y, marker='o', linewidth=1.5, label=model)

    plt.plot([0, 1], [0, 100], '--', color='gray', linewidth=1, label='ideal diagonal (scaled)')
    plt.xlim(0, 1)
    plt.ylim(0, 105)
    plt.xlabel('Orchestrator Confidence Bin Center')
    plt.ylabel('Observed Correct Execution Rate (%)')
    plt.title('Orchestrator Confidence Calibration')
    plt.legend(fontsize=8, ncol=2)
    savefig(OUT / 'fig05_confidence_calibration.png')


def fig6_method_usage_stacked(joined: dict[str, list[dict[str, object]]]) -> None:
    methods = ['i2i', 't2i', 't2t']
    data = np.zeros((len(MODEL_ORDER), len(methods)))

    for i, model in enumerate(MODEL_ORDER):
        recs = joined.get(model, [])
        c = Counter(r['executed_method'] for r in recs)
        total = max(sum(c.values()), 1)
        for j, m in enumerate(methods):
            data[i, j] = 100.0 * c.get(m, 0) / total

    plt.figure(figsize=(8.2, 4.5))
    bottom = np.zeros(len(MODEL_ORDER))
    colors = {'i2i': '#1f77b4', 't2i': '#ff7f0e', 't2t': '#2ca02c'}
    x = np.arange(len(MODEL_ORDER))
    for j, m in enumerate(methods):
        plt.bar(x, data[:, j], bottom=bottom, color=colors[m], label=m)
        bottom += data[:, j]
    plt.xticks(x, MODEL_ORDER)
    plt.ylabel('Share of Queries (%)')
    plt.title('Executed Method Composition by Model')
    plt.legend()
    savefig(OUT / 'fig06_method_usage_stacked.png')


def fig7_web_quality_vs_i2i_gain(joined: dict[str, list[dict[str, object]]]) -> None:
    # x = number of web images; y = i2i - t2i AP on same query.
    plt.figure(figsize=(9.0, 4.8))
    for model in MODEL_ORDER:
        xs, ys = [], []
        for r in joined.get(model, []):
            ai = r['ap_i2i']
            at = r['ap_t2i']
            if not isinstance(ai, float) or not isinstance(at, float) or math.isnan(ai) or math.isnan(at):
                continue
            xs.append(r['num_web_images'])
            ys.append(ai - at)
        if xs:
            plt.scatter(xs, ys, alpha=0.45, s=16, label=model)

    plt.axhline(0, color='black', linewidth=0.9)
    plt.xlabel('Num Web Images (query metadata)')
    plt.ylabel('I2I - T2I AP@50')
    plt.title('Web Image Availability vs I2I Gain')
    plt.legend(fontsize=8, ncol=2)
    savefig(OUT / 'fig07_web_quality_vs_i2i_gain.png')


def fig8_caption_quality_impact(joined: dict[str, list[dict[str, object]]], caption_cache: dict[str, str]) -> None:
    # Proxy: caption text length and failure status distribution, with a separate panel
    # for T2T-proxy AP vs query length.
    caption_texts = list(caption_cache.values())
    lengths = [len(t.split()) for t in caption_texts if t]
    failures = [1 if 'unlabeled image (' in (t or '') else 0 for t in caption_texts]

    # AP proxy by query length from T2T-proxy run.
    proxy_pts = {}
    for model in MODEL_ORDER:
        for r in joined.get(model, []):
            q = r['query']
            if q in proxy_pts:
                continue
            ap = r['ap_t2t_proxy']
            if isinstance(ap, float) and not math.isnan(ap):
                proxy_pts[q] = (len(q.split()), ap)

    qlens = [v[0] for v in proxy_pts.values()]
    qaps = [v[1] for v in proxy_pts.values()]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    ax0, ax1 = axes
    ax0.hist(lengths, bins=20, color='#4c78a8', alpha=0.85)
    fail_rate = 100.0 * (sum(failures) / max(len(failures), 1))
    ax0.set_title(f'Caption Length Distribution\nFailure rate: {fail_rate:.1f}%')
    ax0.set_xlabel('Caption length (tokens)')
    ax0.set_ylabel('Count')

    ax1.scatter(qlens, qaps, alpha=0.5, s=16, color='#f58518')
    ax1.set_title('T2T Proxy AP vs Query Length')
    ax1.set_xlabel('Query length (tokens)')
    ax1.set_ylabel('AP@50 (T2T proxy)')

    savefig(OUT / 'fig08_caption_quality_impact_proxy.png')


def fig9_case_studies(joined: dict[str, list[dict[str, object]]]) -> None:
    # Select queries with largest variance in orchestrator AP across models.
    per_query = defaultdict(dict)
    reasons = {}
    methods = {}
    for model, recs in joined.items():
        for r in recs:
            q = r['query']
            per_query[q][model] = r['ap_orch']
            reasons.setdefault(q, r['router_reason'])
            methods.setdefault(q, r['chosen_method'])

    spread = []
    for q, vals in per_query.items():
        xs = [v for v in vals.values() if isinstance(v, float) and not math.isnan(v)]
        if len(xs) >= 2:
            spread.append((max(xs) - min(xs), q))
    spread.sort(reverse=True)
    picks = [q for _, q in spread[:6]]

    cols = ['Query', 'Chosen', 'AP range across models', 'Reason snippet']
    rows = []
    for q in picks:
        vals = [per_query[q].get(m, float('nan')) for m in MODEL_ORDER]
        vals = [v for v in vals if not math.isnan(v)]
        rnge = (max(vals) - min(vals)) if vals else float('nan')
        reason = reasons.get(q, '')
        if len(reason) > 95:
            reason = reason[:92] + '...'
        rows.append([q[:52] + ('...' if len(q) > 52 else ''), methods.get(q, ''), f'{rnge:.1f}', reason])

    fig, ax = plt.subplots(figsize=(15, 3.8))
    ax.axis('off')
    table = ax.table(cellText=rows, colLabels=cols, loc='center', cellLoc='left', colLoc='left')
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.45)
    plt.title('Case Studies: High-Variance Queries (Orchestrator)')
    savefig(OUT / 'fig09_case_studies_table.png')


def fig10_decision_matrix(joined: dict[str, list[dict[str, object]]]) -> None:
    methods = ['i2i', 't2i', 't2t']
    mat = np.zeros((3, 3), dtype=float)

    for model in MODEL_ORDER:
        for r in joined.get(model, []):
            cand = {
                'i2i': r['ap_i2i'],
                't2i': r['ap_t2i'],
                't2t': r['ap_t2t_proxy'],
            }
            cand = {k: v for k, v in cand.items() if isinstance(v, float) and not math.isnan(v)}
            if not cand:
                continue
            ideal = max(cand.items(), key=lambda kv: kv[1])[0]
            chosen = r['chosen_method']
            if chosen not in methods:
                continue
            mat[methods.index(ideal), methods.index(chosen)] += 1

    row_sums = mat.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    norm = mat / row_sums * 100.0

    fig, ax = plt.subplots(figsize=(5.8, 5.2))
    im = ax.imshow(norm, cmap='Blues', vmin=0, vmax=100)
    ax.set_xticks(range(3), methods)
    ax.set_yticks(range(3), [f'ideal:{m}' for m in methods])
    ax.set_xlabel('orchestrator chosen method')
    ax.set_title('Orchestrator Decision Matrix vs Ideal Method (%)')
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f'{norm[i, j]:.1f}', ha='center', va='center', fontsize=10)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Row-normalized %')
    savefig(OUT / 'fig10_decision_matrix.png')


def write_notes(joined: dict[str, list[dict[str, object]]]) -> None:
    lines = []
    lines.append('Paper Figure Generation Notes')
    lines.append('')
    lines.append('Data sources:')
    lines.append(f'- Orchestrator per-query CSVs: {ROUTER_DIR}')
    lines.append(f'- T2I fixed runs: {T2I_FILES[0].name}, {T2I_FILES[1].name}')
    lines.append(f'- I2I fixed run: {I2I_FILE}')
    lines.append(f'- T2T proxy run: {T2T_PROXY_FILE}')
    lines.append('')
    lines.append('Important caveat:')
    lines.append('- Per-model fixed T2T retrieval CSVs were not found in this workspace.')
    lines.append('- Figures that require T2T per-query per-model use qwen4b caption run as a cross-model proxy.')
    lines.append('')
    lines.append('Record counts by model:')
    for model in MODEL_ORDER:
        lines.append(f'- {model}: {len(joined.get(model, []))} queries')
    (OUT / 'figure_data_notes.txt').write_text('\n'.join(lines), encoding='utf-8')


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    query_map = load_query_map()
    t2i, i2i, t2t_proxy = load_fixed_method_scores()
    orch = load_orchestrator_rows()
    joined = build_joined_records(query_map, t2i, i2i, t2t_proxy, orch)

    if CAPTION_CACHE.exists():
        caption_cache = json.loads(CAPTION_CACHE.read_text(encoding='utf-8'))
    else:
        caption_cache = {}

    fig1_routing_sankey(joined)
    fig2_winrate_heatmap(joined)
    fig3_delta_ap_by_category(joined)
    fig4_oracle_gap(joined)
    fig5_confidence_calibration(joined)
    fig6_method_usage_stacked(joined)
    fig7_web_quality_vs_i2i_gain(joined)
    fig8_caption_quality_impact(joined, caption_cache)
    fig9_case_studies(joined)
    fig10_decision_matrix(joined)
    write_oracle_results_csv(joined)
    write_notes(joined)

    print(f'Saved figures to {OUT}')


if __name__ == '__main__':
    main()
