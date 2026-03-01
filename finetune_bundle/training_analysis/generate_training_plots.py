#!/usr/bin/env python3
"""
Generación de gráficos de validación del fine-tuning.
=====================================================
Extrae los logs de entrenamiento de los notebooks de Colab y genera gráficos
que demuestran la calidad y convergencia del proceso de fine-tuning.

Gráficos generados:
1. Training Loss curve (por step/epoch)
2. Eval Loss curve (por epoch)
3. Train vs Eval Loss (overfitting check)
4. Learning Rate schedule
5. Gradient Norm evolution
6. Perplexity evolution
7. Resumen con tabla de hiperparámetros y métricas finales

Uso:
    python generate_training_plots.py

Los gráficos se guardan en ./plots/
"""

import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")  # Backend no interactivo
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ─────────────────────────────────────────────────────────────────────
# Configuración de estilo
# ─────────────────────────────────────────────────────────────────────

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "#fafafa",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.2,
})

COLORS = {
    "train": "#2196F3",      # Azul
    "eval": "#F44336",       # Rojo
    "lr": "#4CAF50",         # Verde
    "grad": "#FF9800",       # Naranja
    "perplexity": "#9C27B0", # Púrpura
    "highlight": "#E91E63",  # Rosa
    "gap": "#FFCDD2",        # Rojo claro para zona de gap
}

# ─────────────────────────────────────────────────────────────────────
# Rutas
# ─────────────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = SCRIPT_DIR / "artifacts"
PLOTS_DIR = SCRIPT_DIR / "plots"
FINETUNE_BUNDLE_DIR = SCRIPT_DIR.parent

# Notebooks con logs de entrenamiento
NOTEBOOK_PATHS = [
    FINETUNE_BUNDLE_DIR / "finetune_maestro.ipynb",
    FINETUNE_BUNDLE_DIR / "finetune_maestro_v2.ipynb",
]

# ─────────────────────────────────────────────────────────────────────
# Extracción de logs
# ─────────────────────────────────────────────────────────────────────

def extract_logs_from_notebooks() -> Dict[str, List]:
    """
    Parsea los outputs de los notebooks de Colab para extraer
    las métricas step-by-step del Trainer de HuggingFace.

    El entrenamiento se ejecutó en dos fases:
      - Notebook 1 (finetune_maestro.ipynb): ejecuciones cortas iniciales
        (Cell 3: ~0.03 epochs, Cell 4: ~0.34 epochs)
      - Notebook 2 (finetune_maestro_v2.ipynb): ejecución completa hasta
        epoch 5 con resume_from_checkpoint

    Returns:
        Dict con listas de tuplas (epoch, valor) para cada métrica.
    """
    all_entries = []

    for nb_path in NOTEBOOK_PATHS:
        if not nb_path.exists():
            print(f"  [WARN] Notebook no encontrado: {nb_path}")
            continue

        with open(nb_path, encoding="utf-8") as f:
            nb = json.load(f)

        for cell in nb.get("cells", []):
            for out in cell.get("outputs", []):
                if out.get("output_type") != "stream":
                    continue
                text = "".join(out.get("text", []))

                # Training loss entries
                for m in re.finditer(
                    r"\{'loss': ([\d.]+), "
                    r"'grad_norm': ([\d.]+), "
                    r"'learning_rate': ([\d.eE\-+]+), "
                    r"'epoch': ([\d.]+)\}",
                    text,
                ):
                    all_entries.append({
                        "type": "train",
                        "loss": float(m.group(1)),
                        "grad_norm": float(m.group(2)),
                        "lr": float(m.group(3)),
                        "epoch": float(m.group(4)),
                    })

                # Eval loss entries
                for m in re.finditer(
                    r"\{'eval_loss': ([\d.]+),.*?'epoch': ([\d.]+)\}",
                    text,
                ):
                    all_entries.append({
                        "type": "eval",
                        "eval_loss": float(m.group(1)),
                        "epoch": float(m.group(2)),
                    })

                # Final train summary
                for m in re.finditer(
                    r"\{'train_runtime': [\d.]+,.*?'train_loss': ([\d.]+),.*?'epoch': ([\d.]+)\}",
                    text,
                ):
                    all_entries.append({
                        "type": "train_final",
                        "train_loss_final": float(m.group(1)),
                        "epoch": float(m.group(2)),
                    })

    # Deduplicar por (type, epoch) — quedarnos con el último (del run final)
    seen = {}
    for entry in all_entries:
        key = (entry["type"], round(entry["epoch"], 4))
        seen[key] = entry

    entries = sorted(seen.values(), key=lambda x: x["epoch"])

    # Separar en listas
    train_losses = [(e["epoch"], e["loss"]) for e in entries if e["type"] == "train"]
    eval_losses = [(e["epoch"], e["eval_loss"]) for e in entries if e["type"] == "eval"]
    learning_rates = [(e["epoch"], e["lr"]) for e in entries if "lr" in e]
    grad_norms = [(e["epoch"], e["grad_norm"]) for e in entries if "grad_norm" in e]

    return {
        "train_losses": train_losses,
        "eval_losses": eval_losses,
        "learning_rates": learning_rates,
        "grad_norms": grad_norms,
    }


def load_training_summary() -> Dict:
    """Carga el resumen final del entrenamiento."""
    path = ARTIFACTS_DIR / "training_summary.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_model_config() -> Dict:
    """Carga la configuración del modelo."""
    path = ARTIFACTS_DIR / "config.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────────────
# Gráficos
# ─────────────────────────────────────────────────────────────────────

def _annotate_gap(ax, train_data: list, label="Reanudación\ndesde checkpoint"):
    """Anotación visual para el gap entre las fases de entrenamiento."""
    epochs = [e for e, _ in train_data]
    # Detectar gap grande (salto > 1 epoch)
    for i in range(1, len(epochs)):
        if epochs[i] - epochs[i - 1] > 1.0:
            mid = (epochs[i - 1] + epochs[i]) / 2
            ax.axvspan(epochs[i - 1], epochs[i], alpha=0.08, color="gray")
            ymin, ymax = ax.get_ylim()
            ax.annotate(
                label,
                xy=(mid, (ymin + ymax) / 2),
                fontsize=8,
                ha="center",
                va="center",
                color="gray",
                fontstyle="italic",
            )
            break


def plot_training_loss(logs: Dict, summary: Dict):
    """Gráfico 1: Curva de training loss con anotaciones."""
    fig, ax = plt.subplots(figsize=(10, 5))

    epochs = [e for e, _ in logs["train_losses"]]
    losses = [l for _, l in logs["train_losses"]]

    ax.plot(epochs, losses, color=COLORS["train"], linewidth=1.5,
            marker="o", markersize=3, alpha=0.8, label="Training Loss")

    # Marcar puntos clave
    ax.scatter([epochs[0]], [losses[0]], color=COLORS["highlight"],
               s=80, zorder=5, label=f"Inicio: {losses[0]:.2f}")
    ax.scatter([epochs[-1]], [losses[-1]], color=COLORS["train"],
               s=80, zorder=5, marker="*", label=f"Final: {losses[-1]:.2f}")

    # Final train_loss from summary
    final_loss = summary.get("train_loss")
    if final_loss is not None:
        ax.axhline(y=final_loss, color=COLORS["train"], linestyle=":",
                    alpha=0.5, label=f"Loss promedio final: {final_loss:.4f}")

    _annotate_gap(ax, logs["train_losses"])

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (Cross-Entropy)")
    ax.set_title("Curva de Training Loss — Fine-tuning Maestro-REMI")
    ax.legend(loc="upper right")

    fig.savefig(PLOTS_DIR / "01_training_loss.png")
    plt.close(fig)
    print("  ✓ 01_training_loss.png")


def plot_eval_loss(logs: Dict, summary: Dict):
    """Gráfico 2: Curva de evaluation loss."""
    fig, ax = plt.subplots(figsize=(10, 5))

    epochs = [e for e, _ in logs["eval_losses"]]
    losses = [l for _, l in logs["eval_losses"]]

    ax.plot(epochs, losses, color=COLORS["eval"], linewidth=2,
            marker="s", markersize=6, label="Eval Loss")

    # Anotaciones de valores
    for i, (ep, lo) in enumerate(zip(epochs, losses)):
        offset = 10 if i % 2 == 0 else -15
        ax.annotate(f"{lo:.3f}", (ep, lo), textcoords="offset points",
                     xytext=(0, offset), fontsize=8, ha="center",
                     color=COLORS["eval"])

    # Final eval_loss
    final_eval = summary.get("eval_loss")
    if final_eval is not None:
        ax.axhline(y=final_eval, color=COLORS["eval"], linestyle=":",
                    alpha=0.5, label=f"Eval loss final: {final_eval:.4f}")

    _annotate_gap(ax, logs["eval_losses"])

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Eval Loss (Cross-Entropy)")
    ax.set_title("Curva de Evaluation Loss — Fine-tuning Maestro-REMI")
    ax.legend(loc="upper right")

    fig.savefig(PLOTS_DIR / "02_eval_loss.png")
    plt.close(fig)
    print("  ✓ 02_eval_loss.png")


def plot_train_vs_eval(logs: Dict, summary: Dict):
    """Gráfico 3: Train vs Eval Loss — detección de overfitting."""
    fig, ax = plt.subplots(figsize=(10, 5))

    # Train loss
    tr_epochs = [e for e, _ in logs["train_losses"]]
    tr_losses = [l for _, l in logs["train_losses"]]
    ax.plot(tr_epochs, tr_losses, color=COLORS["train"], linewidth=1.5,
            marker="o", markersize=3, alpha=0.7, label="Train Loss")

    # Eval loss
    ev_epochs = [e for e, _ in logs["eval_losses"]]
    ev_losses = [l for _, l in logs["eval_losses"]]
    ax.plot(ev_epochs, ev_losses, color=COLORS["eval"], linewidth=2,
            marker="s", markersize=6, label="Eval Loss")

    # Zona de gap entre eval y train (si hay datos en la misma zona)
    # Indicar final
    final_train = summary.get("train_loss", tr_losses[-1] if tr_losses else None)
    final_eval = summary.get("eval_loss", ev_losses[-1] if ev_losses else None)

    if final_train is not None and final_eval is not None:
        gap = final_eval - final_train
        ax.annotate(
            f"Gap (eval - train): {gap:.3f}\n"
            f"Ratio: {final_eval / final_train:.2f}x",
            xy=(max(tr_epochs[-1], ev_epochs[-1]) * 0.6, (final_eval + final_train) / 2),
            fontsize=10,
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow",
                      edgecolor="orange", alpha=0.9),
        )

    _annotate_gap(ax, logs["train_losses"])

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (Cross-Entropy)")
    ax.set_title("Train vs Eval Loss — Diagnóstico de Overfitting")
    ax.legend(loc="upper right")

    fig.savefig(PLOTS_DIR / "03_train_vs_eval_loss.png")
    plt.close(fig)
    print("  ✓ 03_train_vs_eval_loss.png")


def plot_learning_rate(logs: Dict):
    """Gráfico 4: Schedule del learning rate."""
    fig, ax = plt.subplots(figsize=(10, 4))

    epochs = [e for e, _ in logs["learning_rates"]]
    lrs = [lr for _, lr in logs["learning_rates"]]

    ax.plot(epochs, lrs, color=COLORS["lr"], linewidth=2)
    ax.fill_between(epochs, 0, lrs, alpha=0.15, color=COLORS["lr"])

    _annotate_gap(ax, logs["learning_rates"])

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.set_title("Learning Rate Schedule (Linear con Warmup)")
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-5, -5))

    fig.savefig(PLOTS_DIR / "04_learning_rate.png")
    plt.close(fig)
    print("  ✓ 04_learning_rate.png")


def plot_gradient_norm(logs: Dict):
    """Gráfico 5: Evolución de la norma del gradiente."""
    fig, ax = plt.subplots(figsize=(10, 4))

    epochs = [e for e, _ in logs["grad_norms"]]
    norms = [n for _, n in logs["grad_norms"]]

    ax.plot(epochs, norms, color=COLORS["grad"], linewidth=1.5,
            marker="o", markersize=3, alpha=0.7, label="Gradient Norm")

    # Media y std
    mean_norm = np.mean(norms)
    std_norm = np.std(norms)
    ax.axhline(y=mean_norm, color=COLORS["grad"], linestyle="--",
                alpha=0.6, label=f"Media: {mean_norm:.2f}")
    ax.axhspan(mean_norm - std_norm, mean_norm + std_norm,
                alpha=0.1, color=COLORS["grad"], label=f"±1 std: {std_norm:.2f}")

    _annotate_gap(ax, logs["grad_norms"])

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Gradient Norm (L2)")
    ax.set_title("Evolución de la Norma del Gradiente")
    ax.legend(loc="upper right")

    fig.savefig(PLOTS_DIR / "05_gradient_norm.png")
    plt.close(fig)
    print("  ✓ 05_gradient_norm.png")


def plot_perplexity(logs: Dict, summary: Dict):
    """Gráfico 6: Evolución de la perplexity (exp(eval_loss))."""
    fig, ax = plt.subplots(figsize=(10, 5))

    ev_epochs = [e for e, _ in logs["eval_losses"]]
    ev_losses = [l for _, l in logs["eval_losses"]]
    perplexities = [math.exp(l) for l in ev_losses]

    ax.plot(ev_epochs, perplexities, color=COLORS["perplexity"],
            linewidth=2, marker="D", markersize=7, label="Perplexity")

    # Anotar cada punto
    for ep, pp in zip(ev_epochs, perplexities):
        ax.annotate(f"{pp:.1f}", (ep, pp), textcoords="offset points",
                     xytext=(0, 12), fontsize=9, ha="center",
                     color=COLORS["perplexity"], fontweight="bold")

    # Perplexity final
    final_pp = summary.get("perplexity")
    if final_pp:
        ax.axhline(y=final_pp, linestyle=":", color=COLORS["perplexity"],
                    alpha=0.5, label=f"Final: {final_pp:.2f}")

    # Referencia: perplexity del modelo base pre-entrenado (sin fine-tuning)
    # El modelo base Maestro-REMI tiene perplexity ~107 en este dataset (estimado)
    initial_pp = perplexities[0] if perplexities else None
    if initial_pp and final_pp:
        reduction_pct = (1 - final_pp / initial_pp) * 100
        ax.annotate(
            f"Reducción: {reduction_pct:.1f}%\n"
            f"({initial_pp:.1f} → {final_pp:.2f})",
            xy=(ev_epochs[-1] * 0.5, (initial_pp + final_pp) / 2),
            fontsize=10, ha="center",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lavender",
                      edgecolor=COLORS["perplexity"], alpha=0.9),
        )

    _annotate_gap(ax, logs["eval_losses"])

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Perplexity (exp(eval_loss))")
    ax.set_title("Evolución de la Perplexity — Calidad del Modelo")
    ax.legend(loc="upper right")

    fig.savefig(PLOTS_DIR / "06_perplexity.png")
    plt.close(fig)
    print("  ✓ 06_perplexity.png")


def plot_summary_dashboard(logs: Dict, summary: Dict, config: Dict):
    """Gráfico 7: Dashboard resumen con métricas finales y tabla de hiperparámetros."""
    fig = plt.figure(figsize=(16, 10))

    # Layout: 2 filas x 3 columnas + área inferior para tabla
    gs = fig.add_gridspec(3, 3, hspace=0.45, wspace=0.35,
                          height_ratios=[1, 1, 0.6])

    # ── Panel 1: Train Loss ──
    ax1 = fig.add_subplot(gs[0, 0])
    tr_ep = [e for e, _ in logs["train_losses"]]
    tr_lo = [l for _, l in logs["train_losses"]]
    ax1.plot(tr_ep, tr_lo, color=COLORS["train"], linewidth=1.5, marker="o", markersize=2)
    ax1.set_title("Training Loss", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=9)
    ax1.set_ylabel("Loss", fontsize=9)

    # ── Panel 2: Eval Loss ──
    ax2 = fig.add_subplot(gs[0, 1])
    ev_ep = [e for e, _ in logs["eval_losses"]]
    ev_lo = [l for _, l in logs["eval_losses"]]
    ax2.plot(ev_ep, ev_lo, color=COLORS["eval"], linewidth=2, marker="s", markersize=4)
    ax2.set_title("Eval Loss", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=9)
    ax2.set_ylabel("Loss", fontsize=9)

    # ── Panel 3: Perplexity ──
    ax3 = fig.add_subplot(gs[0, 2])
    perps = [math.exp(l) for l in ev_lo]
    ax3.plot(ev_ep, perps, color=COLORS["perplexity"], linewidth=2, marker="D", markersize=5)
    ax3.set_title("Perplexity", fontsize=11, fontweight="bold")
    ax3.set_xlabel("Epoch", fontsize=9)
    ax3.set_ylabel("Perplexity", fontsize=9)

    # ── Panel 4: Learning Rate ──
    ax4 = fig.add_subplot(gs[1, 0])
    lr_ep = [e for e, _ in logs["learning_rates"]]
    lr_val = [lr for _, lr in logs["learning_rates"]]
    ax4.plot(lr_ep, lr_val, color=COLORS["lr"], linewidth=1.5)
    ax4.fill_between(lr_ep, 0, lr_val, alpha=0.15, color=COLORS["lr"])
    ax4.set_title("Learning Rate", fontsize=11, fontweight="bold")
    ax4.set_xlabel("Epoch", fontsize=9)
    ax4.set_ylabel("LR", fontsize=9)
    ax4.ticklabel_format(axis="y", style="sci", scilimits=(-5, -5))

    # ── Panel 5: Gradient Norm ──
    ax5 = fig.add_subplot(gs[1, 1])
    gn_ep = [e for e, _ in logs["grad_norms"]]
    gn_val = [n for _, n in logs["grad_norms"]]
    ax5.plot(gn_ep, gn_val, color=COLORS["grad"], linewidth=1.5, marker="o", markersize=2)
    ax5.axhline(y=np.mean(gn_val), color=COLORS["grad"], linestyle="--", alpha=0.6)
    ax5.set_title("Gradient Norm", fontsize=11, fontweight="bold")
    ax5.set_xlabel("Epoch", fontsize=9)
    ax5.set_ylabel("L2 Norm", fontsize=9)

    # ── Panel 6: Train vs Eval ──
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.plot(tr_ep, tr_lo, color=COLORS["train"], linewidth=1.5, alpha=0.7, label="Train")
    ax6.plot(ev_ep, ev_lo, color=COLORS["eval"], linewidth=2, label="Eval")
    ax6.set_title("Train vs Eval", fontsize=11, fontweight="bold")
    ax6.set_xlabel("Epoch", fontsize=9)
    ax6.set_ylabel("Loss", fontsize=9)
    ax6.legend(fontsize=8)

    # ── Panel inferior: Tabla de métricas e hiperparámetros ──
    ax_table = fig.add_subplot(gs[2, :])
    ax_table.axis("off")

    # Construir datos de tabla
    final_pp = summary.get("perplexity", "N/A")
    final_eval = summary.get("eval_loss", "N/A")
    final_train = summary.get("train_loss", "N/A")
    runtime_min = summary.get("train_runtime_sec", 0) / 60

    table_data = [
        ["Modelo base", summary.get("model_name", "N/A"),
         "Train Loss", f"{final_train:.4f}" if isinstance(final_train, float) else final_train,
         "Arquitectura", f"GPT-2 ({config.get('n_layer', '?')}L, {config.get('n_head', '?')}H, {config.get('n_embd', '?')}D)"],
        ["Device", summary.get("device", "N/A"),
         "Eval Loss", f"{final_eval:.4f}" if isinstance(final_eval, float) else final_eval,
         "Vocab", f"{config.get('vocab_size', '?')} tokens"],
        ["Epochs", str(summary.get("num_train_epochs", "N/A")),
         "Perplexity", f"{final_pp:.2f}" if isinstance(final_pp, float) else final_pp,
         "Context", f"{config.get('n_positions', '?')} tokens"],
        ["Learning Rate", str(summary.get("learning_rate", "N/A")),
         "Tiempo", f"{runtime_min:.1f} min",
         "Precision", "FP16" if summary.get("used_fp16") else ("BF16" if summary.get("used_bf16") else "FP32")],
        ["Batch (effective)", f"{summary.get('per_device_train_batch_size', '?')} × {summary.get('gradient_accumulation_steps', '?')} = {summary.get('per_device_train_batch_size', 0) * summary.get('gradient_accumulation_steps', 0)}",
         "Train examples", str(summary.get("train_examples", "N/A")),
         "Val examples", str(summary.get("val_examples", "N/A"))],
    ]

    col_labels = ["Hiperparámetro", "Valor", "Métrica", "Valor", "Modelo", "Detalle"]
    table = ax_table.table(
        cellText=table_data,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.4)

    # Colorear encabezados
    for j in range(len(col_labels)):
        table[0, j].set_facecolor("#1565C0")
        table[0, j].set_text_props(color="white", fontweight="bold")
    # Colorear celdas alternas
    for i in range(1, len(table_data) + 1):
        for j in range(len(col_labels)):
            if i % 2 == 0:
                table[i, j].set_facecolor("#E3F2FD")

    fig.suptitle(
        "Dashboard de Fine-tuning — Maestro-REMI → VA Conditioning",
        fontsize=16, fontweight="bold", y=0.98,
    )

    fig.savefig(PLOTS_DIR / "07_dashboard_resumen.png")
    plt.close(fig)
    print("  ✓ 07_dashboard_resumen.png")


def plot_convergence_analysis(logs: Dict, summary: Dict):
    """
    Gráfico 8: Análisis de convergencia.
    Demuestra que el modelo convergió adecuadamente:
    - Loss decreciente y estabilizada
    - Gap train/eval razonable
    - Reducción de perplexity significativa
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # ── Panel A: Convergencia de loss (zoom en últimas epochs) ──
    ax = axes[0]
    # Solo datos de las últimas epochs (fase final, epoch > 4)
    final_train = [(e, l) for e, l in logs["train_losses"] if e > 4.0]
    final_eval = [(e, l) for e, l in logs["eval_losses"] if e > 4.0]

    if final_train:
        ep_tr = [e for e, _ in final_train]
        lo_tr = [l for _, l in final_train]
        ax.plot(ep_tr, lo_tr, color=COLORS["train"], marker="o", markersize=5,
                linewidth=2, label="Train Loss")

        # Línea de tendencia
        z = np.polyfit(ep_tr, lo_tr, 1)
        poly = np.poly1d(z)
        ax.plot(ep_tr, poly(ep_tr), "--", color=COLORS["train"], alpha=0.4,
                label=f"Tendencia: slope={z[0]:.4f}")

    if final_eval:
        ep_ev = [e for e, _ in final_eval]
        lo_ev = [l for _, l in final_eval]
        ax.plot(ep_ev, lo_ev, color=COLORS["eval"], marker="s", markersize=7,
                linewidth=2, label="Eval Loss")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("A) Convergencia (Últimas Epochs)", fontweight="bold")
    ax.legend(fontsize=9)

    # ── Panel B: Barras comparativas de perplexity ──
    ax = axes[1]
    ev_losses = [l for _, l in logs["eval_losses"]]
    perps = [math.exp(l) for l in ev_losses]
    ev_epochs = [e for e, _ in logs["eval_losses"]]

    # Perplexity inicial (primera eval) vs final
    if len(perps) >= 2:
        pp_initial = perps[0]
        pp_final = perps[-1]
        labels = ["Inicial\n(época ~0)", "Final\n(época ~5)"]
        values = [pp_initial, pp_final]
        colors_bar = [COLORS["highlight"], COLORS["perplexity"]]

        bars = ax.bar(labels, values, color=colors_bar, width=0.5, edgecolor="gray")

        # Anotar valores
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f"{val:.1f}", ha="center", va="bottom", fontweight="bold", fontsize=12)

        # Flecha de reducción
        reduction = (1 - pp_final / pp_initial) * 100
        ax.annotate(
            f"↓ {reduction:.1f}%",
            xy=(0.5, (pp_initial + pp_final) / 2),
            fontsize=16, ha="center", fontweight="bold",
            color=COLORS["lr"],
        )

    ax.set_ylabel("Perplexity")
    ax.set_title("B) Reducción de Perplexity", fontweight="bold")

    # ── Panel C: Gap train/eval (indicador de overfitting) ──
    ax = axes[2]

    # Calcular gaps en los puntos de eval
    eval_data = list(zip(ev_epochs, ev_losses))
    gaps = []
    gap_epochs = []

    for ev_ep, ev_loss in eval_data:
        # Encontrar el train loss más cercano a esta epoch
        closest_train = None
        min_dist = float("inf")
        for tr_ep, tr_loss in logs["train_losses"]:
            dist = abs(tr_ep - ev_ep)
            if dist < min_dist:
                min_dist = dist
                closest_train = tr_loss
        if closest_train is not None and min_dist < 0.5:
            gaps.append(ev_loss - closest_train)
            gap_epochs.append(ev_ep)

    if gaps:
        colors_gap = [COLORS["lr"] if g < 1.0 else COLORS["grad"] if g < 2.0 else COLORS["eval"]
                       for g in gaps]
        ax.bar([str(round(e, 2)) for e in gap_epochs], gaps, color=colors_gap,
               edgecolor="gray", width=0.6)

        # Línea de referencia
        ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, label="Gap = 1.0")

        # Anotar valores
        for i, (ep_str, g) in enumerate(zip([str(round(e, 2)) for e in gap_epochs], gaps)):
            ax.text(i, g + 0.02, f"{g:.2f}", ha="center", fontsize=9, fontweight="bold")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Gap (Eval - Train)")
    ax.set_title("C) Gap Train/Eval (Overfitting)", fontweight="bold")
    if gaps:
        ax.legend(fontsize=9)

    fig.suptitle(
        "Análisis de Convergencia del Fine-tuning",
        fontsize=14, fontweight="bold", y=1.02,
    )

    fig.savefig(PLOTS_DIR / "08_convergence_analysis.png")
    plt.close(fig)
    print("  ✓ 08_convergence_analysis.png")


def plot_full_training_history(logs: Dict, summary: Dict):
    """
    Gráfico 9: Historia completa de entrenamiento con fases anotadas.
    Muestra las dos fases de entrenamiento (notebook 1 + notebook 2)
    y la reanudación desde checkpoint.
    """
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(12, 8),
                                          gridspec_kw={"height_ratios": [2, 1]})

    tr_ep = [e for e, _ in logs["train_losses"]]
    tr_lo = [l for _, l in logs["train_losses"]]
    ev_ep = [e for e, _ in logs["eval_losses"]]
    ev_lo = [l for _, l in logs["eval_losses"]]

    # ── Top: Loss completo ──
    ax_top.plot(tr_ep, tr_lo, color=COLORS["train"], linewidth=1.5,
                marker="o", markersize=3, alpha=0.8, label="Train Loss")
    ax_top.plot(ev_ep, ev_lo, color=COLORS["eval"], linewidth=2,
                marker="s", markersize=6, label="Eval Loss")

    # Fases
    # Fase 1: epoch 0 - 0.34 (notebooks iniciales)
    # Fase 2: epoch 4.43 - 5.0 (notebook final con resume)
    ax_top.axvspan(0, 0.35, alpha=0.06, color="blue", label="Fase 1: exploración")
    ax_top.axvspan(0.35, 4.4, alpha=0.04, color="gray")
    ax_top.axvspan(4.4, 5.1, alpha=0.06, color="green", label="Fase 2: continuación")

    ax_top.annotate("Fase 1\n(exploración en Colab)", xy=(0.17, max(tr_lo) * 0.85),
                     fontsize=9, ha="center", color="blue", fontstyle="italic")
    ax_top.annotate("Fase 2\n(resume from checkpoint\nhasta epoch 5)",
                     xy=(4.7, max(tr_lo) * 0.5),
                     fontsize=9, ha="center", color="green", fontstyle="italic")
    ax_top.annotate("Entrenamiento intermedio\n(logs no capturados en notebook,\n"
                     "checkpoints guardados)",
                     xy=(2.4, max(tr_lo) * 0.65),
                     fontsize=8, ha="center", color="gray", fontstyle="italic")

    ax_top.set_xlabel("Epoch")
    ax_top.set_ylabel("Loss (Cross-Entropy)")
    ax_top.set_title("Historia Completa del Entrenamiento", fontweight="bold")
    ax_top.legend(loc="upper right", fontsize=9)

    # ── Bottom: Learning rate + grad norm ──
    color_lr = COLORS["lr"]
    ax_bot.plot([e for e, _ in logs["learning_rates"]],
                [lr for _, lr in logs["learning_rates"]],
                color=color_lr, linewidth=1.5, label="Learning Rate")
    ax_bot.set_xlabel("Epoch")
    ax_bot.set_ylabel("Learning Rate", color=color_lr)
    ax_bot.tick_params(axis="y", labelcolor=color_lr)
    ax_bot.ticklabel_format(axis="y", style="sci", scilimits=(-5, -5))

    # Eje secundario para gradient norm
    ax_gn = ax_bot.twinx()
    ax_gn.plot([e for e, _ in logs["grad_norms"]],
               [n for _, n in logs["grad_norms"]],
               color=COLORS["grad"], linewidth=1, alpha=0.6, label="Grad Norm")
    ax_gn.set_ylabel("Gradient Norm", color=COLORS["grad"])
    ax_gn.tick_params(axis="y", labelcolor=COLORS["grad"])

    # Fases en bottom
    ax_bot.axvspan(0, 0.35, alpha=0.06, color="blue")
    ax_bot.axvspan(4.4, 5.1, alpha=0.06, color="green")

    # Leyenda combinada
    lines1, labels1 = ax_bot.get_legend_handles_labels()
    lines2, labels2 = ax_gn.get_legend_handles_labels()
    ax_bot.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=9)

    ax_bot.set_title("Learning Rate y Gradient Norm", fontweight="bold")

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "09_full_training_history.png")
    plt.close(fig)
    print("  ✓ 09_full_training_history.png")


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("GENERACIÓN DE GRÁFICOS DE VALIDACIÓN DEL FINE-TUNING")
    print("=" * 70)

    # Crear directorio de plots
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Extraer logs
    print("\n[1/3] Extrayendo logs de entrenamiento desde notebooks...")
    logs = extract_logs_from_notebooks()
    print(f"  Train loss entries: {len(logs['train_losses'])}")
    print(f"  Eval loss entries:  {len(logs['eval_losses'])}")
    print(f"  LR entries:         {len(logs['learning_rates'])}")
    print(f"  Grad norm entries:  {len(logs['grad_norms'])}")

    if not logs["train_losses"]:
        print("\n[ERROR] No se encontraron logs de entrenamiento.")
        print("  Verifica que los notebooks con outputs existan en:")
        for p in NOTEBOOK_PATHS:
            print(f"    {p}")
        sys.exit(1)

    # 2. Cargar artifacts
    print("\n[2/3] Cargando artifacts del modelo...")
    summary = load_training_summary()
    config = load_model_config()
    print(f"  Modelo: {summary.get('model_name')}")
    print(f"  Train loss final: {summary.get('train_loss')}")
    print(f"  Eval loss final:  {summary.get('eval_loss')}")
    print(f"  Perplexity:       {summary.get('perplexity')}")

    # Guardar logs extraídos como JSON (para referencia/reproducibilidad)
    logs_path = ARTIFACTS_DIR / "extracted_training_logs.json"
    with open(logs_path, "w", encoding="utf-8") as f:
        json.dump(logs, f, indent=2, ensure_ascii=False)
    print(f"  Logs guardados en: {logs_path}")

    # 3. Generar gráficos
    print(f"\n[3/3] Generando gráficos en {PLOTS_DIR}/")

    plot_training_loss(logs, summary)
    plot_eval_loss(logs, summary)
    plot_train_vs_eval(logs, summary)
    plot_learning_rate(logs)
    plot_gradient_norm(logs)
    plot_perplexity(logs, summary)
    plot_summary_dashboard(logs, summary, config)
    plot_convergence_analysis(logs, summary)
    plot_full_training_history(logs, summary)

    # Resumen
    print("\n" + "=" * 70)
    print("GRÁFICOS GENERADOS EXITOSAMENTE")
    print("=" * 70)
    plots = sorted(PLOTS_DIR.glob("*.png"))
    for p in plots:
        size_kb = p.stat().st_size / 1024
        print(f"  {p.name:45s} ({size_kb:.0f} KB)")
    print(f"\nTotal: {len(plots)} gráficos en {PLOTS_DIR}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
