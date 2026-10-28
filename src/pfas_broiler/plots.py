"""Figures of the article. Each function draws one figure and returns it."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import FormatStrFormatter, FuncFormatter  # noqa: E402

from . import data  # noqa: E402
from .config import EU_MAXIMUM_LEVEL, EXPOSURE_STOP_DAY, GROUPS, LOQ, MATRIX_TO_COMPARTMENT  # noqa: E402
from .growth import GOMPERTZ_PARAMETERS, body_weight_g  # noqa: E402

STYLE = {
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "axes.linewidth": 1.0,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "mathtext.fontset": "custom",  # math text in the same font as the labels
    "mathtext.rm": "Arial",
    "mathtext.it": "Arial:italic",
    "mathtext.bf": "Arial:bold",
    "mathtext.sf": "Arial",
    "mathtext.cal": "Arial",
    "pdf.fonttype": 42,  # embed TrueType fonts in vector output
}
GROUP_COLORS = {g: f"C{i}" for i, g in enumerate(GROUPS)}
TISSUE_COLORS = {"plasma": "C3", "liver": "C1", "kidney": "C2", "muscle": "C0"}
TISSUE_LABELS = {"plasma": "Serum", "liver": "Liver", "kidney": "Kidney", "muscle": "Muscle"}
_LOQ_HALF = LOQ / 2.0  # below-LOQ values are DRAWN at LOQ/2 (display only)


def _box(ax, text, x=0.04, y=0.96, **kwargs):
    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox=dict(
            boxstyle="round,pad=0.4", facecolor="white", edgecolor="0.7", linewidth=0.8, alpha=0.95
        ),
        **kwargs,
    )


def growth_curve() -> plt.Figure:
    """Body weight per group (mean ± SEM), Gompertz fit and Ross 308 reference."""
    plt.rcParams.update(STYLE)
    weights = data.body_weights().dropna(subset=["day", "body_weight_g"])
    fig, ax = plt.subplots(figsize=(6.5, 4.2))

    for group in sorted(weights["group"].unique()):
        agg = (
            weights[weights["group"] == group].groupby("day")["body_weight_g"].agg(["mean", "sem"])
        )
        color = GROUP_COLORS.get(group, "gray")
        ax.scatter(
            agg.index,
            agg["mean"],
            c=color,
            label=group,
            s=60,
            alpha=0.85,
            edgecolors="white",
            linewidths=0.6,
            zorder=2,
        )
        ax.errorbar(
            agg.index,
            agg["mean"],
            yerr=agg["sem"].fillna(0),
            fmt="none",
            color=color,
            alpha=0.6,
            capsize=2,
        )

    days = np.linspace(0, 36, 200)
    ax.plot(
        days, [body_weight_g(d) for d in days], color="k", lw=1.8, label="Gompertz fit", zorder=1
    )
    ross = data.ross308_body_weight()
    ross = ross[ross["day"] <= 36]
    ax.plot(
        ross["day"],
        ross["body_weight_g"],
        color="0.45",
        ls="--",
        lw=1.4,
        label="Ross 308 male",
        zorder=1,
    )

    for group, stop in sorted(EXPOSURE_STOP_DAY.items(), key=lambda kv: kv[1]):
        if group == "E5":
            continue  # exposed until slaughter
        ax.axvline(stop, color=GROUP_COLORS[group], ls=":", lw=1.2, zorder=0)
        ax.text(
            stop,
            0.03,
            f" {group}",
            transform=ax.get_xaxis_transform(),
            color=GROUP_COLORS[group],
            fontsize=9,
            ha="left",
            va="bottom",
        )
    ax.plot([], [], color="0.3", ls=":", lw=1.2, label="Feed switch")

    a, k, t0 = GOMPERTZ_PARAMETERS
    _box(
        ax,
        "\n".join(
            [
                r"$W(t) = A\,\exp\!\left(-\exp\!\left(-C\,(t - t_0)\right)\right)$",
                rf"$A$ = {a:.0f} g,  $C$ = {k:.3f} d$^{{-1}}$,  $t_0$ = {t0:.1f} d",
                rf"$R^2$ = {growth_r2():.3f}",
            ]
        ),
        x=0.02,
        y=0.98,
        fontsize=10,
        linespacing=1.5,
    )

    ax.set_xlabel("Day")
    ax.set_ylabel("Body weight (g)")
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(0.01, 0.76),
        framealpha=0.95,
        handlelength=1.8,
        labelspacing=0.35,
        borderaxespad=0.0,
    )
    ax.grid(True, alpha=0.3, lw=0.8)
    ax.set_xlim(left=0)
    fig.tight_layout()
    return fig


def growth_r2() -> float:
    """R2 of the Gompertz curve against the individual body weights (g)."""
    weights = data.body_weights().dropna(subset=["day", "body_weight_g"])
    observed = weights["body_weight_g"].to_numpy()
    predicted = np.array([body_weight_g(d) for d in weights["day"]])
    residual = np.sum((observed - predicted) ** 2)
    return float(1.0 - residual / np.sum((observed - observed.mean()) ** 2))


def observed_vs_predicted(tables: dict[str, pd.DataFrame], fit: dict[str, dict]) -> plt.Figure:
    """log10 predicted vs observed for several compounds (marker = compound).

    tables: compound -> output of goodness_of_fit.observed_vs_predicted;
    fit: compound -> output of goodness_of_fit.metrics.
    """
    plt.rcParams.update(STYLE)
    markers = dict(zip(tables, ["^", "o", "s", "D", "v"], strict=False))
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    lo, hi = np.log10(_LOQ_HALF), -np.inf

    for compound, table in tables.items():
        detected = table[~table["censored"]]
        censored = table[table["censored"]]
        x, y = np.log10(detected["observed"]), np.log10(detected["predicted"])
        y_cens = np.log10(censored["predicted"])
        lo = min(lo, x.min(), y.min(), y_cens.min() if len(y_cens) else np.inf)
        hi = max(hi, x.max(), y.max(), y_cens.max() if len(y_cens) else -np.inf)
        for tissue, color in TISSUE_COLORS.items():
            sel = detected["compartment"] == tissue
            ax.scatter(
                x[sel],
                y[sel],
                c=color,
                marker=markers[compound],
                s=60,
                alpha=0.85,
                edgecolors="white",
                linewidths=0.6,
            )
            sel = censored["compartment"] == tissue
            ax.scatter(
                [np.log10(_LOQ_HALF)] * sel.sum(),
                y_cens[sel],
                facecolors="none",
                edgecolors=color,
                marker=markers[compound],
                s=60,
                linewidths=1.1,
            )

    pad = 0.05 * (hi - lo)
    line = np.array([lo - pad, hi + pad])
    ax.fill_between(
        line, line - np.log10(2), line + np.log10(2), color="gray", alpha=0.15, zorder=0
    )
    ax.plot(line, line, "k--", alpha=0.7, lw=1.8)
    ax.set_xlim(*line)
    ax.set_ylim(*line)
    ax.set_xlabel("log10(observed concentration)")
    ax.set_ylabel("log10(predicted concentration, posterior median)")
    ax.grid(True, alpha=0.3, lw=0.8)
    ax.set_aspect("equal", adjustable="box")
    _box(
        ax,
        "\n".join(
            f"{c}: R²={m['r2']:.3f}, GMFE={m['gmfe']:.3f}, Bias={m['bias_log10']:.3f}"
            for c, m in fit.items()
        ),
        fontsize=9,
    )

    handles = [
        Line2D([0], [0], marker="o", ls="", color=c, label=TISSUE_LABELS[t], markersize=8)
        for t, c in TISSUE_COLORS.items()
    ]
    handles += [
        Line2D([0], [0], marker=m, ls="", color="gray", label=c, markersize=8)
        for c, m in markers.items()
    ]
    handles += [
        Line2D([0], [0], ls="--", color="k", label="1:1", lw=1.8),
        Patch(facecolor="gray", alpha=0.15, label="2-fold"),
        Line2D(
            [0],
            [0],
            marker="o",
            ls="",
            markerfacecolor="none",
            markeredgecolor="0.4",
            label="< LOQ (at LOQ/2)",
            markersize=8,
        ),
    ]
    ax.legend(handles=handles, loc="lower right", framealpha=0.95)
    fig.tight_layout()
    return fig


def concentration_time(compound: str, predictions: pd.DataFrame) -> plt.Figure:
    """Predicted (posterior median) and measured concentrations over time, per tissue."""
    plt.rcParams.update(STYLE)
    measured = data.tissue_concentrations()
    measured = measured[(measured["compound"] == compound) & (measured["isomer"] == "Linear")]
    measured = measured.assign(compartment=measured["matrix"].map(MATRIX_TO_COMPARTMENT))
    detected = measured[measured["concentration"] > 0]
    cell_max = measured.groupby(["group", "day", "compartment"], as_index=False)[
        "concentration"
    ].max()
    censored = cell_max[cell_max["concentration"] <= 0]
    limits = EU_MAXIMUM_LEVEL.get(compound, {})

    fig, axes = plt.subplots(2, 2, figsize=(9, 7), sharex=True)
    for ax, tissue in zip(axes.ravel(), TISSUE_LABELS, strict=True):
        for group in GROUPS:
            color = GROUP_COLORS[group]
            pred = predictions[
                (predictions["group"] == group) & (predictions["compartment"] == tissue)
            ]
            ax.plot(pred["day"], pred["median"], color=color, lw=1.8)
            d = detected[(detected["group"] == group) & (detected["compartment"] == tissue)]
            ax.scatter(
                d["day"],
                d["concentration"],
                color=color,
                s=24,
                alpha=0.55,
                edgecolor="none",
                zorder=4,
            )
            c = censored[(censored["group"] == group) & (censored["compartment"] == tissue)]
            ax.scatter(
                c["day"],
                [_LOQ_HALF] * len(c),
                facecolors="none",
                edgecolors=color,
                s=30,
                linewidths=1.1,
                zorder=4,
            )
        if tissue in limits:
            ml = limits[tissue]
            ax.axhline(ml, color="0.25", ls="--", lw=1.3, zorder=4)
            ax.text(
                0.97,
                ml,
                f"ML {ml:g} µg kg$^{{-1}}$",
                transform=ax.get_yaxis_transform(),
                va="bottom",
                ha="right",
                fontsize=8,
                color="0.25",
            )
        _box(ax, TISSUE_LABELS[tissue], x=0.03, fontsize=11)
        unit = "µg L$^{-1}$" if tissue == "plasma" else "µg kg$^{-1}$"
        ax.set_ylabel(f"Concentration ({unit})")
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        ax.grid(True, alpha=0.3, lw=0.8)
        ax.margins(y=0.12)
        ax.set_xlim(left=0)
    for ax in axes[-1]:
        ax.set_xlabel("Age (days)")

    groups = [
        Line2D(
            [0],
            [0],
            color=GROUP_COLORS[g],
            lw=2.5,
            marker="o",
            markeredgecolor="none",
            markersize=6,
            label=g,
        )
        for g in GROUPS
    ]
    fig.add_artist(
        fig.legend(
            handles=groups,
            title="Exposure group",
            loc="lower center",
            ncol=5,
            bbox_to_anchor=(0.5, 0.055),
        )
    )
    key = [
        Line2D([0], [0], color="0.4", lw=2.0, label="model median"),
        Line2D(
            [0],
            [0],
            color="0.4",
            marker="o",
            ls="none",
            alpha=0.6,
            markersize=6,
            label="animal (detected)",
        ),
        Line2D(
            [0],
            [0],
            color="0.4",
            marker="o",
            ls="none",
            markerfacecolor="none",
            markeredgecolor="0.4",
            markersize=7,
            label="< LOQ (plotted at LOQ/2)",
        ),
        Line2D([0], [0], color="0.25", ls="--", label="EU maximum level"),
    ]
    fig.legend(
        handles=key,
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 0.0),
        fontsize=8,
    )
    fig.tight_layout(rect=[0, 0.12, 1, 1])
    return fig


def _plain_log_labels(axis) -> None:
    """Label log-axis ticks as plain numbers (0.1, 2, 3) instead of 1e-1, 2x10^0."""

    def minor(value, _):  # label only 2, 3 and 5 between decades, to avoid crowding
        leading = round(value / 10 ** np.floor(np.log10(value)))
        return f"{value:g}" if leading in (2, 3, 5) else ""

    axis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    axis.set_minor_formatter(FuncFormatter(minor))


def posterior(post) -> plt.Figure:
    """Joint posterior of (k_elim, P_il) with both marginals; one curve for one parameter.

    post: a :class:`pfas_broiler.inference.posterior.GridPosterior`.
    """
    plt.rcParams.update(STYLE)
    tex = {"k_elim": r"$k_{\mathrm{elim}}\ (\mathrm{d}^{-1})$", "P_il": r"$P_{\mathrm{IL}}$"}
    probs = [0.025, 0.5, 0.975]
    styles = [":", "-", ":"]

    def view(name):  # crop to the region that holds the mass
        lo, hi = np.log10(post.quantiles(name, [0.0005, 0.9995]))
        pad = max(0.5 * (hi - lo), 0.4)
        return 10 ** (lo - pad), 10 ** (hi + pad)

    if len(post.names) == 1:
        name = post.names[0]
        fig, ax = plt.subplots(figsize=(7.0, 5.0))
        values, mass = post.marginal(name)
        ax.plot(values, mass, color="C0", lw=2)
        for q, ls in zip(post.quantiles(name, probs), styles, strict=True):
            ax.axvline(q, color="C3", lw=1.2, ls=ls)
        ax.set_xscale("log")
        _plain_log_labels(ax.xaxis)
        ax.set_xlim(*view(name))
        ax.set_xlabel(tex[name])
        ax.set_ylabel("posterior probability")
        return fig

    y_name, x_name = post.names  # (k_elim, P_il): P_il on the x axis
    fig = plt.figure(figsize=(7.2, 6.8))
    gs = GridSpec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4], wspace=0.06, hspace=0.06)
    ax = fig.add_subplot(gs[1, 0])
    top = fig.add_subplot(gs[0, 0], sharex=ax)
    right = fig.add_subplot(gs[1, 1], sharey=ax)

    x, _ = post.marginal(x_name)
    y, _ = post.marginal(y_name)
    density = post.weights / post.weights.max()  # axes (k_elim, P_il)
    mesh = ax.pcolormesh(x, y, density, shading="auto", cmap="viridis")
    ax.set_xscale("log")
    ax.set_yscale("log")
    _plain_log_labels(ax.xaxis)
    _plain_log_labels(ax.yaxis)
    ax.set_xlabel(tex[x_name])
    ax.set_ylabel(tex[y_name])
    ax.text(0.04, 0.04, "joint posterior", transform=ax.transAxes, color="white", fontsize=10)

    top.plot(*post.marginal(x_name), color="C0", lw=1.6)
    values, mass = post.marginal(y_name)
    right.plot(mass, values, color="C0", lw=1.6)
    for q, ls in zip(post.quantiles(x_name, probs), styles, strict=True):
        ax.axvline(q, color="white", lw=1.0, ls=ls, alpha=0.8)
        top.axvline(q, color="C3", lw=1.0, ls=ls)
    for q, ls in zip(post.quantiles(y_name, probs), styles, strict=True):
        ax.axhline(q, color="white", lw=1.0, ls=ls, alpha=0.8)
        right.axhline(q, color="C3", lw=1.0, ls=ls)
    top.set_ylabel(f"marginal\n{tex[x_name]}", fontsize=10)
    right.set_xlabel(f"marginal\n{tex[y_name]}", fontsize=10)
    top.tick_params(axis="x", which="both", labelbottom=False)
    right.tick_params(axis="y", which="both", labelleft=False)
    top.set_yticks([])
    right.set_xticks([])

    bar = fig.colorbar(
        mesh,
        cax=ax.inset_axes([0.56, 0.90, 0.40, 0.022]),
        orientation="horizontal",
        ticks=[0, 0.5, 1.0],
    )
    bar.set_label("relative posterior density", color="white", fontsize=8, labelpad=2)
    bar.ax.tick_params(labelsize=7, colors="white", length=2)
    bar.outline.set_edgecolor("white")
    ax.set_xlim(*view(x_name))
    ax.set_ylim(*view(y_name))
    return fig
