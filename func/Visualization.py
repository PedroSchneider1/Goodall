import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from sklearn.metrics import roc_curve

def _save_visualizations(results_df, records_full, output_dir, timestamp, timings):
    """
    records_full: list of dicts, each containing:
        - all keys from results_df
        - 'y_true': np.ndarray  (ground truth per peak)
        - 'y_prob': np.ndarray  (predicted probabilities per peak)
        - 'false_negatives': int
    """

    PALETTE = {
        'xgboost':       "#1C578A",
        'random_forest': "#1F7A47",
        'cnn1d':         "#B86127",
        'rnn':           "#712E8B",
    }
    FONT_TITLE  = {'fontsize': 13, 'fontweight': 'bold', 'color': '#1a1a2e'}
    FONT_LABEL  = {'fontsize': 10, 'color': '#444'}
    FONT_TICK   = {'labelsize': 9, 'colors': '#555'}
    BG          = '#F7F9FC'
    GRID_COLOR  = '#DDE3EC'

    models  = results_df.index.tolist()
    colors  = [PALETTE.get(m, '#888') for m in models]

    def _style_ax(ax, title):
        ax.set_facecolor(BG)
        ax.set_title(title, **FONT_TITLE, pad=10)
        ax.tick_params(**FONT_TICK)
        ax.grid(axis='y', color=GRID_COLOR, linewidth=0.8, zorder=0)
        ax.spines[['top', 'right']].set_visible(False)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontsize(9)
            label.set_color('#555')


    # ========================================
    # Feature extraction timing
    # ========================================
    timing_rows = []

    for phase, info in timings.items():
        if not isinstance(info, dict):
            continue
        per_file_vals = list(info.get("per_file", {}).values())

        total_time = info.get("elapsed_s", 0)
        mean_time = np.mean(per_file_vals) if per_file_vals else 0
        median_time = np.median(per_file_vals) if per_file_vals else 0
        n_files = len(per_file_vals)

        timing_rows.append([
            phase,
            f"{total_time:.2f}s",
            f"{mean_time:.4f}s",
            f"{median_time:.4f}s",
            str(n_files)
        ])

    # Table
    fig, ax = plt.subplots(figsize=(9, 2 + len(timing_rows)*0.6), facecolor='white')
    ax.axis('off')

    col_labels = ['Phase', 'Total', 'Mean/File', 'Median/File', 'Files']

    tbl = ax.table(
        cellText=timing_rows,
        colLabels=col_labels,
        cellLoc='center',
        loc='center'
    )

    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.8)

    # Header style
    for j in range(len(col_labels)):
        cell = tbl[0, j]
        cell.set_facecolor('#1a1a2e')
        cell.set_text_props(color='white', fontweight='bold')
        cell.set_edgecolor('white')

    # Body style
    for i in range(1, len(timing_rows)+1):
        for j in range(len(col_labels)):
            cell = tbl[i, j]
            cell.set_facecolor('#F7F9FC' if i % 2 == 0 else 'white')
            cell.set_edgecolor('#DDE3EC')

    ax.set_title('Feature extraction timing summary', **FONT_TITLE, pad=15)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/timing_summary_table.png', dpi=180, bbox_inches='tight')
    plt.close()

    # Graph
    phases = [row[0] for row in timing_rows]
    totals = [float(row[1].replace('s', '')) for row in timing_rows]

    fig, ax = plt.subplots(figsize=(7, 4), facecolor='white')

    bars = ax.bar(
        phases,
        totals,
        color=['#2D6A9F', '#E07B39'],
        width=0.5,
        edgecolor='white',
        linewidth=0.8
    )

    for bar, val in zip(bars, totals):
        ax.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.5,
            f'{val:.2f}s',
            ha='center',
            fontsize=9
        )

    _style_ax(ax, 'Total execution time by phase')
    ax.set_ylabel('Seconds', **FONT_LABEL)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/timing_summary_graph.png', dpi=180, bbox_inches='tight')
    plt.close()

    # ========================================
    # Training time
    # ========================================
    fig, ax = plt.subplots(figsize=(7, 4), facecolor='white')
    bars = ax.bar(models, results_df['training_time_s'], color=colors,
                  width=0.5, zorder=3, edgecolor='white', linewidth=0.8)
    for bar, val in zip(bars, results_df['training_time_s']):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f'{val:.1f}s', ha='center', va='bottom', fontsize=9, color='#333')
    _style_ax(ax, 'Training time (seconds)')
    ax.set_ylabel('Seconds', **FONT_LABEL)
    ax.set_xlabel('Model', **FONT_LABEL)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/training_time.png', dpi=150, bbox_inches='tight')
    plt.close()

    # ========================================
    # ROC curves
    # (one per model, all on same axes)
    # ========================================
    fig, ax = plt.subplots(figsize=(7, 6), facecolor='white')
    ax.set_facecolor(BG)
    ax.plot([0, 1], [0, 1], '--', color='#aaa', linewidth=1, label='Random (AUC = 0.50)')
    for rec in records_full:
        fpr, tpr, _ = roc_curve(rec['y_true'], rec['y_prob'])
        auc_val      = rec['roc_auc']
        model_name   = rec['model']
        ax.plot(fpr, tpr, color=PALETTE.get(model_name, '#888'),
                linewidth=2.2, label=f"{model_name}  (AUC = {auc_val:.4f})")
    ax.set_xlabel('False positive rate', **FONT_LABEL)
    ax.set_ylabel('True positive rate',  **FONT_LABEL)
    ax.set_title('ROC curves — all models', **FONT_TITLE, pad=10)
    ax.legend(fontsize=9, framealpha=0.9, edgecolor=GRID_COLOR)
    ax.grid(color=GRID_COLOR, linewidth=0.8)
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/roc_curves.png', dpi=150, bbox_inches='tight')
    plt.close()

    # ========================================
    # False negatives comparison
    # ========================================
    fn_vals = [rec['false_negatives'] for rec in records_full]
    fig, ax  = plt.subplots(figsize=(7, 4), facecolor='white')
    bars     = ax.bar(models, fn_vals, color=colors, width=0.5,
                      zorder=3, edgecolor='white', linewidth=0.8)
    for bar, val in zip(bars, fn_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                str(val), ha='center', va='bottom', fontsize=10,
                fontweight='bold', color='#c0392b')
    _style_ax(ax, 'False negatives per model  (lower = better)')
    ax.set_ylabel('Count', **FONT_LABEL)
    ax.set_xlabel('Model', **FONT_LABEL)
    # Highlight the best (minimum)
    best_idx = int(np.argmin(fn_vals))
    bars[best_idx].set_edgecolor('#c0392b')
    bars[best_idx].set_linewidth(2)
    ax.annotate('fewest FN', xy=(best_idx, fn_vals[best_idx]),
                xytext=(best_idx + 0.3, fn_vals[best_idx] + max(fn_vals) * 0.08),
                fontsize=8, color='#c0392b',
                arrowprops=dict(arrowstyle='->', color='#c0392b', lw=1.2))
    plt.tight_layout()
    plt.savefig(f'{output_dir}/false_negatives.png', dpi=150, bbox_inches='tight')
    plt.close()

    # ========================================
    # Classification metrics grouped bar chart
    # ========================================
    metrics  = ['precision', 'sensitivity', 'specificity', 'f1', 'roc_auc']
    n_models = len(models)
    n_metrics = len(metrics)
    x        = np.arange(n_metrics)
    width    = 0.18

    fig, ax  = plt.subplots(figsize=(11, 5), facecolor='white')
    ax.set_facecolor(BG)
    for i, (model, color) in enumerate(zip(models, colors)):
        vals   = [results_df.loc[model, m] for m in metrics]
        offset = (i - n_models / 2 + 0.5) * width
        bars   = ax.bar(x + offset, vals, width, color=color,
                        label=model, zorder=3, edgecolor='white', linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics], fontsize=10)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel('Score', **FONT_LABEL)
    ax.set_title('Classification metrics — all models', **FONT_TITLE, pad=10)
    ax.legend(fontsize=9, framealpha=0.9, edgecolor=GRID_COLOR)
    ax.grid(axis='y', color=GRID_COLOR, linewidth=0.8, zorder=0)
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/metrics_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

    # ========================================
    # DataFrame as publication-ready table
    # ========================================
    display_df = results_df[['training_time_s', 'precision', 'sensitivity',
                              'specificity', 'f1', 'roc_auc']].copy()
    display_df.columns = ['Train time (s)', 'Precision', 'Sensitivity',
                          'Specificity', 'F1', 'ROC-AUC']
    # Format floats
    for col in display_df.columns[1:]:
        display_df[col] = display_df[col].map('{:.4f}'.format)
    display_df['Train time (s)'] = display_df['Train time (s)'].map('{:.2f}s'.format)

    fig, ax = plt.subplots(figsize=(11, 2.2 + len(models) * 0.55), facecolor='white')
    ax.axis('off')

    col_labels = ['Model'] + display_df.columns.tolist()
    cell_text  = [[m] + display_df.loc[m].tolist() for m in models]

    tbl = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellLoc='center',
        loc='center',
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.8)

    # Header styling
    for j in range(len(col_labels)):
        cell = tbl[0, j]
        cell.set_facecolor('#1a1a2e')
        cell.set_text_props(color='white', fontweight='bold', fontsize=10)
        cell.set_edgecolor('white')

    # Row styling — alternating + highlight best FN row
    best_fn_model = models[int(np.argmin(fn_vals))]
    for i, model in enumerate(models, start=1):
        row_color = '#EAF0FB' if i % 2 == 0 else 'white'
        if model == best_fn_model:
            row_color = '#FFF3CD'   # highlight lowest-FN model
        for j in range(len(col_labels)):
            cell = tbl[i, j]
            cell.set_facecolor(row_color)
            cell.set_edgecolor('#DDE3EC')
            if j == 0:
                cell.set_text_props(fontweight='bold', color=PALETTE.get(model, '#333'))

    legend_patch = mpatches.Patch(color='#FFF3CD', label='Fewest false negatives')
    ax.legend(handles=[legend_patch], loc='lower right',
              fontsize=8, framealpha=0.9, edgecolor='#DDE3EC')

    ax.set_title(f'Model comparison results — {timestamp}',
                 **FONT_TITLE, pad=16, loc='left')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/results_table.png', dpi=200, bbox_inches='tight')
    plt.close()

    print(f"Visualizations saved to '{output_dir}/'")