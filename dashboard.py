import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from pathlib import Path
import io

# Import your local modules
from utils.read_results import ResultsReader
from engine.strategies import AxisScale, PlotStrategy, PlotContext
from engine.colors import ColorManagerFactory

st.set_page_config(layout="wide", page_title="Optimizer Results Dashboard")

# --- Helper: Data Loading ---
@st.cache_resource
def load_data(filepath):
    """Load and cache the results reader."""
    return ResultsReader(filepath)

def get_available_npz_files():
    """Recursively find all results.npz files."""
    return list(Path(".").rglob("results.npz"))

# --- Helper: Exact Legend Replica from plotting.py ---
def add_custom_split_legend(ax, color_manager, lrs, rhos, show_styles=False):
    """
    Replicates the exact legend logic from plotting.py.
    Uses Patches for headers and Line2D for entries.
    """
    legend_elements = []

    # 1. Split Style Indicators (Optional)
    if show_styles:
        legend_elements.append(Line2D([0], [0], color="black", linestyle="-", linewidth=2, label="SAM (Solid)"))
        legend_elements.append(Line2D([0], [0], color="black", linestyle="--", linewidth=2, alpha=0.5, label="Base (Dashed)"))
        legend_elements.append(Patch(facecolor="none", edgecolor="none", label="   ")) # Spacer

    # 2. Learning Rate (Hue)
    if lrs:
        legend_elements.append(Patch(facecolor="none", edgecolor="none", label="Learning Rate (Hue):"))
        for lr in lrs:
            # Use bright color (rho=0 equivalent or max vibrancy) for legend
            c = color_manager.color_lr(lr) 
            legend_elements.append(Line2D([0], [0], color=c, linewidth=3, label=f"  lr={lr}"))

    # 3. Rho (Vibrancy)
    # Only show if we have rhos > 0 or multiple rhos
    valid_rhos = sorted(list(set(rhos)))
    if len(valid_rhos) > 1 or (len(valid_rhos) == 1 and valid_rhos[0] > 0):
        legend_elements.append(Patch(facecolor="none", edgecolor="none", label="   ")) # Spacer
        legend_elements.append(Patch(facecolor="none", edgecolor="none", label="rho (Vibrancy, SAM only):"))
        
        # Use a sample LR to show rho effect
        sample_lr = lrs[0] if lrs else 0.1
        
        for rho in valid_rhos:
            # Compute color config
            c = color_manager.color_config(sample_lr, rho)
            legend_elements.append(Line2D([0], [0], color=c, linewidth=3, alpha=0.8, label=f"  rho={rho}"))

    # Configure the legend above the plot
    # Dynamic ncol calculation to keep it readable
    total_items = len(legend_elements)
    ncols = min(total_items, 8) 
    
    # Place legend above the plot (y=1.02)
    ax.legend(
        handles=legend_elements, 
        loc='lower center', 
        bbox_to_anchor=(0.5, 1.02),
        ncol=ncols, 
        frameon=True,
        fontsize='small',
        handletextpad=0.2,
        columnspacing=1.0,
        edgecolor='lightgray'
    )

# --- Sidebar: Configuration ---
st.sidebar.title("Experiment Config")

# File Selection
available_files = get_available_npz_files()
if not available_files:
    st.error("No 'results.npz' files found in current directory.")
    st.stop()

selected_file = st.sidebar.selectbox(
    "Select Results File", 
    available_files, 
    format_func=lambda x: str(x)
)

try:
    reader = load_data(str(selected_file))
except Exception as e:
    st.error(f"Error loading file: {e}")
    st.stop()

# Global Data Parsing
all_optimizers = reader.optimizers
all_metrics = reader.metrics
all_params = reader.hyperparams

# --- Sidebar: Filtering ---
st.sidebar.header("Global Filters")
selected_opts = st.sidebar.multiselect("Select Optimizers", all_optimizers, default=all_optimizers)

# Extract LRs and Rhos for Color Management
all_lrs = set()
all_rhos = set()
for opt in selected_opts:
    for p in all_params.get(opt, []):
        if 'lr' in p: all_lrs.add(p['lr'])
        if 'rho' in p: all_rhos.add(p['rho'])

sorted_lrs = sorted(list(all_lrs))
sorted_rhos = sorted(list(all_rhos))

# --- Main Interface ---
st.title("Experiment Findings Explorer")
st.markdown(f"**Loaded:** `{selected_file}`")

tabs = st.tabs(["Trajectory Analysis (Finding 1)", "SAM vs Base (Finding 2)", "Hyperparam Grid (Finding 3)", "Stability Analysis"])

# ==============================================================================
# TAB 1: General Trajectory Analysis
# ==============================================================================
with tabs[0]:
    st.header("Trajectory Analysis")
    
    col1, col2 = st.columns(2)
    with col1:
        plot_metric = st.selectbox("Metric (Y-Axis)", all_metrics, index=all_metrics.index("angle") if "angle" in all_metrics else 0)
    with col2:
        x_axis_type = st.radio("X-Axis Scale", ["Log", "Linear"], horizontal=True, key="t1_x")

    strategy = PlotStrategy(
        x_scale=AxisScale.Log if x_axis_type == "Log" else AxisScale.Linear,
        y_scale=AxisScale.Log if "loss" in plot_metric or "distance" in plot_metric or "angle" in plot_metric else AxisScale.Linear
    )
    
    fig, ax = plt.subplots(figsize=(12, 7))
    strategy.configure_axis(ax, base_label=plot_metric)
    
    colors = ColorManagerFactory.create_husl_manager(sorted_lrs, sorted_rhos)
    
    count = 0
    for opt in selected_opts:
        opt_params = all_params.get(opt, [])
        for params in opt_params:
            lr = params.get('lr')
            rho = params.get('rho', 0.0)
            
            run_data = []
            steps = None
            for seed in reader.seeds:
                try:
                    data = reader.get_data(opt, params, seed, plot_metric)
                    steps_data = reader.get_data(opt, params, seed, 'steps')
                    run_data.append(data)
                    steps = steps_data
                except KeyError: pass
            
            if run_data and steps is not None:
                mean_data = np.mean(np.stack(run_data), axis=0)
                c = colors.color_config(lr, rho)
                
                ctx = PlotContext(
                    ax=ax, x=steps, y=mean_data,
                    label="_nolegend_", 
                    plot_kwargs={"color": c, "linewidth": 1.5, "alpha": 0.8}
                )
                strategy.plot(ctx)
                count += 1

    if count > 0:
        add_custom_split_legend(ax, colors, sorted_lrs, sorted_rhos)
        st.pyplot(fig)
        
        fn = f"{plot_metric}_trajectory.pdf"
        img = io.BytesIO()
        fig.savefig(img, format='pdf', bbox_inches='tight')
        st.download_button(label="Download PDF", data=img, file_name=fn, mime="application/pdf")

# ==============================================================================
# TAB 2: SAM vs Base (Finding 2)
# ==============================================================================
with tabs[1]:
    st.header("Finding 2: SAM vs Base Comparison")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        base_opts = [o for o in all_optimizers if "SAM" not in o]
        base_opt_select = st.selectbox("Base Optimizer", base_opts, index=0 if base_opts else 0)
    with col2:
        sam_opts = [o for o in all_optimizers if "SAM" in o]
        sam_opt_select = st.selectbox("SAM Variant", sam_opts, index=0 if sam_opts else 0)
    with col3:
        metric_select = st.selectbox("Metric", [m for m in all_metrics], index=0)
    
    fig2, ax2 = plt.subplots(figsize=(12, 7))
    strategy2 = PlotStrategy(
        x_scale=AxisScale.Log, 
        y_scale=AxisScale.Log if "loss" in metric_select or "angle" in metric_select else AxisScale.Linear
    )
    strategy2.configure_axis(ax2, base_label=metric_select)
    
    colors = ColorManagerFactory.create_husl_manager(sorted_lrs, sorted_rhos)
    
    # Plot Base
    for params in all_params.get(base_opt_select, []):
        lr = params.get('lr')
        for seed in reader.seeds:
            try:
                y = reader.get_data(base_opt_select, params, seed, metric_select)
                x = reader.get_data(base_opt_select, params, seed, 'steps')
                c = colors.color_config(lr, rho=0.0)
                ax2.plot(x, y, color=c, linestyle='--', alpha=0.6, linewidth=1.5, label="_nolegend_")
                break
            except: pass

    # Plot SAM
    for params in all_params.get(sam_opt_select, []):
        lr = params.get('lr')
        rho = params.get('rho', 0.0)
        for seed in reader.seeds:
            try:
                y = reader.get_data(sam_opt_select, params, seed, metric_select)
                x = reader.get_data(sam_opt_select, params, seed, 'steps')
                c = colors.color_config(lr, rho)
                ax2.plot(x, y, color=c, linestyle='-', alpha=0.9, linewidth=2.0, label="_nolegend_")
                break
            except: pass

    add_custom_split_legend(ax2, colors, sorted_lrs, sorted_rhos, show_styles=True)
    st.pyplot(fig2)
    
    fn2 = f"{base_opt_select}_vs_{sam_opt_select}.pdf"
    img2 = io.BytesIO()
    fig2.savefig(img2, format='pdf', bbox_inches='tight')
    st.download_button("Download Plot PDF", data=img2, file_name=fn2, mime="application/pdf")

# ==============================================================================
# TAB 3: Hyperparam Grid (Finding 3)
# ==============================================================================
with tabs[2]:
    st.header("Finding 3: Hyperparameter Grid")
    
    grid_metric = st.selectbox("Grid Metric", all_metrics, index=all_metrics.index("angle") if "angle" in all_metrics else 0, key="t3_metric")
    grid_lrs = st.multiselect("Grid Learning Rates", sorted_lrs, default=sorted_lrs[:4] if len(sorted_lrs) > 4 else sorted_lrs)
    grid_rhos = st.multiselect("Grid Rhos", sorted_rhos, default=[r for r in sorted_rhos if r > 0])
    
    if grid_lrs and grid_rhos:
        nrows, ncols = len(grid_rhos), len(grid_lrs)
        fig3, axes3 = plt.subplots(nrows, ncols, figsize=(3.5*ncols, 3*nrows), sharex=True, sharey=True, constrained_layout=True)
        
        if nrows == 1 and ncols == 1: axes3 = np.array([[axes3]])
        elif nrows == 1: axes3 = axes3[np.newaxis, :]
        elif ncols == 1: axes3 = axes3[:, np.newaxis]
        
        color_strat = ColorManagerFactory.create_paired_optimizer_manager(selected_opts, grid_rhos)

        for i, rho in enumerate(grid_rhos):
            for j, lr in enumerate(grid_lrs):
                ax = axes3[i, j]
                ax.set_xscale('log')
                ax.set_yscale('log')
                ax.grid(True, which='both', alpha=0.2)
                
                # Base Optimizers
                for opt in [o for o in selected_opts if "SAM" not in o]:
                    params = next((p for p in all_params.get(opt, []) if p['lr'] == lr), None)
                    if params:
                        try:
                            y = reader.get_data(opt, params, 0, grid_metric)
                            x = reader.get_data(opt, params, 0, 'steps')
                            c = color_strat.color_config(opt, rho=0)
                            ax.plot(x, y, color=c, alpha=0.5, linestyle='--', linewidth=1.5)
                        except: pass

                # SAM Optimizers
                for opt in [o for o in selected_opts if "SAM" in o]:
                    params = next((p for p in all_params.get(opt, []) if p['lr'] == lr and p.get('rho') == rho), None)
                    if params:
                        try:
                            y = reader.get_data(opt, params, 0, grid_metric)
                            x = reader.get_data(opt, params, 0, 'steps')
                            c = color_strat.color_config(opt, rho=rho)
                            ax.plot(x, y, color=c, alpha=0.9, linestyle='-', linewidth=2.0)
                        except: pass
                
                if i == 0: ax.set_title(f"LR = {lr}", fontsize=10)
                if j == 0: ax.set_ylabel(f"Rho = {rho}\n{grid_metric}", fontsize=9)
        
        # Grid Legend
        legend_elements = []
        for opt in selected_opts:
            c = color_strat.color_config(opt, rho=0.1)
            style = '-' if "SAM" in opt else '--'
            legend_elements.append(Line2D([0], [0], color=c, lw=2, linestyle=style, label=opt))
        fig3.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=len(selected_opts))

        st.pyplot(fig3)
        img3 = io.BytesIO()
        fig3.savefig(img3, format='pdf', bbox_inches='tight')
        st.download_button("Download Grid PDF", data=img3, file_name="hyperparam_grid.pdf", mime="application/pdf")

# ==============================================================================
# TAB 4: Stability Analysis
# ==============================================================================
with tabs[3]:
    st.header("Stability Analysis")
    stab_metric = st.selectbox("Stability Metric", ["w_norm", "update_norm", "grad_norm"], index=0)
    
    available_stab = [m for m in all_metrics if stab_metric in m]
    if available_stab:
        metric_key = available_stab[0]
        fig4, ax4 = plt.subplots(figsize=(12, 7))
        strategy4 = PlotStrategy(x_scale=AxisScale.Log, y_scale=AxisScale.Linear)
        strategy4.configure_axis(ax4, base_label=metric_key)
        
        colors = ColorManagerFactory.create_sequential_lr_manager(sorted_lrs, sorted_rhos)
        
        for opt in selected_opts:
            for params in all_params.get(opt, []):
                lr = params.get('lr')
                rho = params.get('rho', 0.0)
                try:
                    y = reader.get_data(opt, params, 0, metric_key)
                    x = reader.get_data(opt, params, 0, 'steps')
                    c = colors.color_config(lr, rho)
                    ax4.plot(x, y, color=c, label="_nolegend_")
                except: pass
        
        add_custom_split_legend(ax4, colors, sorted_lrs, sorted_rhos)
        st.pyplot(fig4)
