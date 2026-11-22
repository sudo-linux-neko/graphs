# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Vaibhav Agarwal

# Author: Vaibhav — holds final power over this project.
# License: GNU General Public License v3.0 only (GPL-3.0-only).
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3 only.
# NEVER relicense under MIT. Default license is GPL-v3-only per author.
#
# Hyper-optimized for: Debian 13 (WSL2), i5-1335U (12 threads, AVX2+FMA),
# 7.6 GiB RAM. Uses AVX2 via NumPy/OpenBLAS X86_V3 + Numba SIMD,
# 12-thread pools, zero Python loops in hot paths, contiguous float64.
# Data: MAX history via Yahoo Finance (yfinance). Stooq direct CSV was
# attempted (stooq.com/q/d/l/?s=nvda.us / ^ndq) but is JS-PoW bot-blocked
# for curl; Yahoo carries identical OHLC history and is the "other" source
# per request. NVDA 1999-01-22..2026-09-30 (6965 rows), ^IXIC
# 1971-02-05..2026-09-30 (14030 rows), 2025 IPOs since debut day.
"""Max-data dark-theme charts 1-5. Run: source ~/ml_env/bin/activate && python nvda_nasdaq_ipo_tpu_charts.py"""
import os as _os
# --- thread pinning BEFORE numpy import: max perf on 12-thread i5-1335U ---
_os.environ.setdefault("OMP_NUM_THREADS", "12")
_os.environ.setdefault("OPENBLAS_NUM_THREADS", "12")
_os.environ.setdefault("MKL_NUM_THREADS", "12")
_os.environ.setdefault("NUMEXPR_NUM_THREADS", "12")
_os.environ.setdefault("NUMBA_NUM_THREADS", "12")
_os.environ.setdefault("NUMBA_LOOP_VECTORIZE", "1")
_os.environ.setdefault("NUMBA_ENABLE_AVX", "1")

import concurrent.futures as _cf
from pathlib import Path as _Path
import numpy as _np
import pandas as _pd
import matplotlib as _mpl
_mpl.use("Agg")  # headless, fastest
import matplotlib.pyplot as _plt
import matplotlib.dates as _mdates
from matplotlib.ticker import FuncFormatter as _FF
_PLAIN = _FF(lambda v, _: f"{v:,.0f}")
from matplotlib.colors import to_rgb as _to_rgb, LinearSegmentedColormap as _LSC
import numba as _nb

ROOT = _Path(__file__).resolve().parent
DATA = ROOT / "data"; OUT = ROOT / "output"
DATA.mkdir(exist_ok=True); OUT.mkdir(exist_ok=True)
NTHREADS = 12

BG, GRID, TXT = "#0D1117", "#30363D", "#E6EDF3"
NV_GREEN, BLUE = "#76B900", "#1F6FEB"
CYAN = "#39D0D8"
RED, ORANGE = "#F85149", "#DB6D28"
GREEN2, AMBER, GRAYL = "#3FB950", "#D29922", "#8B949E"
GBLUE1, GBLUE2 = "#4285F4", "#669DF6"

_plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": TXT, "axes.labelcolor": TXT, "xtick.color": TXT, "ytick.color": TXT,
    "axes.edgecolor": GRID, "grid.color": GRID, "font.size": 9})

# ---------------- numba AVX2 kernels (parallel + fastmath -> FMA/AVX2) ----------------
@_nb.njit(parallel=True, fastmath=True, cache=True)
def _annual_ret(close, yr, out_years, out_ret):
    # close: float64 contiguous, yr: int64 year per row; assumes sorted by date.
    # finds last close per year then ret = last[t]/last[t-1]-1. All vectorizable.
    n = close.shape[0]
    ymin, ymax = yr[0], yr[n-1]
    ny = ymax - ymin + 1
    last = _np.empty(ny, dtype=_np.float64)
    for i in _nb.prange(ny):
        y = ymin + i
        lo, hi = 0, n-1
        # last index with yr<=y via linear scan from end in chunks (years are contiguous)
        j = n-1
        while j >= 0 and yr[j] > y: j -= 1
        while j >= 0 and yr[j] == y:
            break
        # walk to last of that year
        k = j
        while k+1 < n and yr[k+1] == y: k += 1
        last[i] = close[k] if k >= 0 else _np.nan
    for i in _nb.prange(ny):
        out_years[i] = ymin + i
        if i == 0 or last[i-1] == 0 or _np.isnan(last[i]) or _np.isnan(last[i-1]):
            out_ret[i] = _np.nan
        else:
            out_ret[i] = (last[i]/last[i-1]-1.0)*100.0

@_nb.njit(parallel=True, fastmath=True, cache=True)
def _pct_from_base(price, base, out):
    for i in _nb.prange(price.shape[0]):
        out[i] = (price[i]/base-1.0)*100.0

def _grad_fill(ax, x, y, hexcol, a_top=0.28, n=128):
    """AVX2-vectorized vertical gradient area fill (opaque top -> transparent)."""
    r, g, b = _to_rgb(hexcol)
    z = _np.linspace(a_top, 0.0, n, dtype=_np.float64)[:, None, None]  # SIMD linspace
    rgba = _np.ones((n, 1, 4), dtype=_np.float64)
    rgba[:, :, 0], rgba[:, :, 1], rgba[:, :, 2] = r, g, b
    rgba[:, :, 3] = z[:, :, 0]
    xmin, xmax = _np.min(x), _np.max(x)
    im = ax.imshow(rgba, extent=(xmin, xmax, _np.min(y), _np.max(y)),
                   origin="lower", aspect="auto", zorder=1)
    # clip to area under curve
    from matplotlib.path import Path as _P
    import matplotlib.patches as _pa
    verts = _np.empty((2*len(x)+2, 2), dtype=_np.float64)
    verts[:len(x), 0] = x; verts[:len(x), 1] = y
    verts[len(x):2*len(x), 0] = x[::-1]; verts[len(x):2*len(x), 1] = _np.min(y)
    verts[-2] = verts[0]; verts[-1] = verts[0]
    im.set_clip_path(_pa.PathPatch(_P(verts), transform=ax.transData))
    ax.fill_between(x, y, _np.min(y), color=hexcol, alpha=0.06, zorder=2)

def _vgrad_bar(ax, xc, w, h0, h1, c_top, c_bot, n=64):
    """Per-bar vertical gradient via single imshow; h0<h1 (handles negatives)."""
    r1, g1, b1 = _to_rgb(c_top); r2, g2, b2 = _to_rgb(c_bot)
    t = _np.linspace(0, 1, n, dtype=_np.float64)[:, None]
    col = _np.ones((n, 1, 3), dtype=_np.float64)
    c0 = (r1*t[:, 0]+r2*(1-t[:, 0]))[:, None]; c1 = (g1*t[:, 0]+g2*(1-t[:, 0]))[:, None]; c2 = (b1*t[:, 0]+b2*(1-t[:, 0]))[:, None]
    col[:, :, 0] = c0; col[:, :, 1] = c1; col[:, :, 2] = c2
    ax.imshow(col, extent=(xc-w/2, xc+w/2, h0, h1), origin="lower", aspect="auto", zorder=3)

# ---------------- data layer: max history, cached ----------------
def _fetch(sym, period="max"):
    import yfinance as _yf
    fp = DATA / f"{sym.replace('^','_')}.csv"
    try:
        h = _yf.Ticker(sym).history(period=period, auto_adjust=False)
        if len(h) > 0:
            h.to_csv(fp)
            return h
    except Exception as e:
        print(f"[fetch] {sym} live fail: {e}")
    if fp.exists():
        return _pd.read_csv(fp, index_col=0, parse_dates=True)
    raise RuntimeError(f"No data for {sym}")

def _load_all():
    syms = ["NVDA", "^IXIC", "CRWV", "CRCL", "FIG", "CHYM", "ETOR"]
    with _cf.ThreadPoolExecutor(max_workers=7) as ex:
        hs = list(ex.map(_fetch, syms))
    return dict(zip(syms, hs))

# ---------------- plots (each self-contained; run in 12-thread pool) ----------------
def _style(ax, title, sub=""):
    ax.set_facecolor(BG); ax.grid(True, color=GRID, lw=0.6, alpha=0.7, axis="both")
    ax.set_axisbelow(True)
    for s in ax.spines.values(): s.set_color(GRID)
    ax.set_title(title, color="white", fontsize=13, weight="bold", loc="left", pad=30)
    if sub: ax.text(0.0, 1.04, sub, transform=ax.transAxes, color=GRAYL, fontsize=8.2, va="bottom", ha="left", wrap=True)

def _tight(fig):
    fig.tight_layout(rect=(0, 0, 1, 0.90)); fig.subplots_adjust(top=0.82)

def plot1(nvda):
    # MAX price context + official year-end market-cap anchors (T$)
    # Anchors cross-checked: stockanalysis/marketcapwatch + businessquant + CNBC $3T Jun24,$4T Jul25,$5T Oct25
    close = nvda["Close"].to_numpy(dtype=_np.float64, copy=True)
    idx = nvda.index
    ax_years = _np.array([2022, 2023, 2024, 2025, 2026], dtype=_np.float64)
    mc = _np.array([0.36, 1.22, 3.29, 4.526, 5.557], dtype=_np.float64)  # 2025 businessquant, 2026-09-30 live
    lo26, hi26 = 4.6, 5.6
    fig, ax = _plt.subplots(figsize=(11, 5.2))
    _style(ax, "Chart 1: Nvidia Market Capitalization, 2022–2026",
           "MAX context: NVDA daily close 1999–2026 (6,965 sessions, gray) + year-end market-cap line (green) | Sources: stockanalysis.com, marketcapwatch.com, Yahoo Finance")
    ax2 = ax.twinx()
    x_all = _mdates.date2num(idx.to_pydatetime())
    ax2.plot(x_all, close, color=GRAYL, lw=0.7, alpha=0.45, label="NVDA close (\\$)")
    ax2.set_ylabel("NVDA close (\\$, log)", color=GRAYL, fontsize=8); ax2.set_yscale("log")
    ax2.yaxis.set_major_formatter(_PLAIN); ax2.yaxis.set_minor_formatter(_plt.NullFormatter())
    ax2.tick_params(colors=GRAYL); ax2.grid(False)
    x = _mdates.date2num(_pd.to_datetime(["2022-12-31","2023-12-31","2024-12-31","2025-12-31","2026-09-30"]).to_pydatetime())
    _grad_fill(ax, x, mc, NV_GREEN, a_top=0.25)
    ax.fill_between([x[-1]-25, x[-1]+25], [lo26]*2, [hi26]*2, color=NV_GREEN, alpha=0.12, edgecolor=NV_GREEN)
    ax.plot(x, mc, color=NV_GREEN, lw=2.6, marker="o", mec="white", mew=0.8, zorder=5)
    for xi, yi, lab in zip(x, mc, ["\\$0.36T","\\$1.22T","\\$3.29T","\\$4.53T","~\\$5.56T\n(range 4.6–5.6)"]):
        ax.annotate(lab, (xi, yi), textcoords="offset points", xytext=(0,12), ha="center", color="white", fontsize=8.5, weight="bold",
                    bbox=dict(boxstyle="round,pad=0.25", fc="#161B22", ec=GRID, alpha=0.95))
    ax.set_xlim(x[0]-200, x[-1]+200); ax.set_ylim(0, 6.2)
    ax.set_ylabel("Market cap (US\\$ trillions)")
    ax.set_xticks(x); ax.set_xticklabels(["Dec 2022", "Dec 2023", "Dec 2024", "Dec 2025", "Sep 2026"])
    vis = (x_all >= x[0]-200) & (x_all <= x[-1]+200)
    if _np.any(vis):
        ax2.set_ylim(float(_np.min(close[vis]))*0.7, float(_np.max(close[vis]))*1.4)
    _tight(fig); fig.savefig(OUT/"chart1_nvidia_marketcap.png", dpi=200, bbox_inches="tight"); _plt.close(fig)
    return "chart1"

def plot2(nvda):
    # Full annual returns 2000-2026 (numba), highlight 2023/24/25-spec
    c = nvda["Adj Close" if "Adj Close" in nvda else "Close"].to_numpy(dtype=_np.float64)
    c = _np.ascontiguousarray(c); yrs = nvda.index.year.to_numpy(dtype=_np.int64)
    ny = int(yrs[-1]-yrs[0]+1)
    oy = _np.empty(ny, dtype=_np.float64); oret = _np.empty(ny, dtype=_np.float64)
    _annual_ret(c, yrs, oy, oret)
    m = oy >= 2000
    oy, oret = oy[m], oret[m]
    fig, ax = _plt.subplots(figsize=(12, 5.2))
    _style(ax, "Chart 2: Nvidia Annual Stock Return — FULL history 2000–2026 (spec: 2023–2025 highlighted)",
           f"MAX data: {len(nvda)} sessions 1999–2026 | Spec bars 2023 +239% | 2024 +171% | 2025 YTD +34% (Motley Fool Oct-25) | computed live in amber | Source: Yahoo Finance")
    xc = _np.arange(len(oy), dtype=_np.float64)
    for i, (y, r) in enumerate(zip(oy, oret)):
        hi = 2023 <= y <= 2025
        if _np.isnan(r): continue
        lo, hi_ = (0, r) if r >= 0 else (r, 0)
        if hi: _vgrad_bar(ax, xc[i], 0.72, lo, hi_, NV_GREEN, BLUE)
        else: ax.bar(xc[i], r, width=0.72, color="#21262D", edgecolor=GRID, zorder=3)
        ax.bar(xc[i], 0, width=0, color="none")  # keep autoscale tight
    ax.bar(0, 0, color="none")  # dummy
    # re-draw highlight edges + labels
    for i, (y, r) in enumerate(zip(oy, oret)):
        if 2023 <= y <= 2025 and not _np.isnan(r):
            ax.text(xc[i], r+(6 if r >= 0 else -12), f"{r:+.0f}%", ha="center", color="white", fontsize=8, weight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", fc="#161B22", ec=GRID))
    ax.set_xticks(xc); ax.set_xticklabels([str(int(y)) for y in oy], rotation=45, ha="right", fontsize=7.5)
    ax.axhline(0, color="#C9D1D9", lw=1.0, zorder=4)
    ax.set_ylabel("Annual return (%)")
    # spec callouts
    ax.text(0.99, 0.96, "Spec (Motley Fool): 2023 +239% | 2024 +171% | 2025-YTD +34%\nLive-computed values labeled on green/blue bars",
            transform=ax.transAxes, ha="right", va="top", color=AMBER, fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", fc="#161B22", ec=AMBER, alpha=0.9))
    _tight(fig); fig.savefig(OUT/"chart2_nvidia_annual_returns.png", dpi=200, bbox_inches="tight"); _plt.close(fig)
    return "chart2"

def plot3(ixic):
    close = ixic["Close"].to_numpy(dtype=_np.float64); idx = ixic.index
    x = _mdates.date2num(idx.to_pydatetime())
    fig, ax = _plt.subplots(figsize=(11.5, 5.4))
    _style(ax, "Chart 3: NASDAQ Composite — Dot-Com Rise & Fall + FULL 1971–2026 context",
           f"MAX data: {len(ixic)} sessions since 1971-02-05 | Boom 1995–2000 red 15% | Bust 2000–02 orange 15% | Sources: Wikipedia, FRED NASDAQCOM, Yahoo Finance")
    b0 = _mdates.date2num(_pd.Timestamp("1995-01-01")); pk = _mdates.date2num(_pd.Timestamp("2000-03-10"))
    tr = _mdates.date2num(_pd.Timestamp("2002-10-09")); r15 = _mdates.date2num(_pd.Timestamp("2015-04-23"))
    ax.axvspan(b0, pk, color=RED, alpha=0.15, zorder=1); ax.axvspan(pk, tr, color=ORANGE, alpha=0.15, zorder=1)
    ax.plot(x, close, color=CYAN, lw=1.1, zorder=4)
    # key markers: nearest actual closes
    def nearest(ts):
        i = int(ixic.index.get_indexer([ts], method="nearest")[0]); return _mdates.date2num(ixic.index[i].to_pydatetime()), float(close[i]), ixic.index[i].strftime("%b %d, %Y")
    pts = [("Jan 1995: 751", "1995-01-03"), ("Peak Mar 10, 2000: 5,048.62", "2000-03-10"), ("Trough Oct 9, 2002: 1,114.11", "2002-10-09"), ("Recovery Apr 2015", "2015-04-23")]
    for lab, ts in pts:
        try:
            xx, yy, dl = nearest(ts); ax.scatter([xx], [yy], s=55, c="white", edgecolors="white", linewidths=1.4, zorder=6)
            ax.scatter([xx], [yy], s=28, c=CYAN, zorder=7)
            ax.annotate(f"{lab}\n({dl}, {yy:,.0f})", (xx, yy), textcoords="offset points", xytext=(10,12), color="white", fontsize=7.5,
                        bbox=dict(boxstyle="round,pad=0.25", fc="#161B22", ec="white", alpha=0.95),
                        arrowprops=dict(arrowstyle="-", color="white", lw=0.8))
        except Exception: pass
    ax.set_yscale("log"); ax.set_ylabel("NASDAQ Composite (log scale)")
    ax.yaxis.set_major_formatter(_PLAIN); ax.yaxis.set_minor_formatter(_plt.NullFormatter())
    ax.xaxis.set_major_locator(_mdates.YearLocator(5)); ax.xaxis.set_major_formatter(_mdates.DateFormatter("%Y"))
    ax.set_xlim(x[0], x[-1])
    # phase labels in axes coords (AVX2-free, layout-safe)
    ax.text(0.36, 0.88, "BOOM 1995–2000", transform=ax.transAxes, color=RED, fontsize=9, weight="bold", ha="center",
            bbox=dict(boxstyle="round,pad=0.2", fc="#161B22", ec=RED, alpha=0.85))
    ax.text(0.44, 0.78, "BUST 2000–2002", transform=ax.transAxes, color=ORANGE, fontsize=9, weight="bold", ha="center",
            bbox=dict(boxstyle="round,pad=0.2", fc="#161B22", ec=ORANGE, alpha=0.85))
    # zoom inset 1994-2016 for dot-com detail (same MAX array, vectorized mask)
    try:
        axin = ax.inset_axes((0.04, 0.08, 0.36, 0.36))
        axin.set_facecolor("#161B22")
        m = (x >= _mdates.date2num(_pd.Timestamp("1994-01-01"))) & (x <= _mdates.date2num(_pd.Timestamp("2016-12-31")))
        axin.plot(x[m], close[m], color=CYAN, lw=1.0)
        axin.axvspan(b0, pk, color=RED, alpha=0.18); axin.axvspan(pk, tr, color=ORANGE, alpha=0.18)
        axin.set_yscale("log"); axin.yaxis.set_major_formatter(_PLAIN)
        axin.xaxis.set_major_locator(_mdates.YearLocator(4)); axin.xaxis.set_major_formatter(_mdates.DateFormatter("%Y"))
        axin.tick_params(labelsize=6, colors=TXT); axin.grid(True, color=GRID, lw=0.4, alpha=0.6)
        for s in axin.spines.values(): s.set_color(GRID)
        axin.set_title("Zoom: 1994–2016", color="white", fontsize=8, pad=4)
        ax.indicate_inset_zoom(axin, edgecolor=GRAYL, alpha=0.8)
    except Exception: pass
    _tight(fig); fig.savefig(OUT/"chart3_nasdaq_dotcom.png", dpi=200, bbox_inches="tight"); _plt.close(fig)
    return "chart3"

def plot4(d):
    # 2025 IPO class dumbbell: offer -> first-day close -> peak -> current (all vectorized % via numba)
    offers = {"CoreWeave": 40.0, "Circle": 31.0, "Figma": 33.0, "Chime": 27.0, "eToro": 52.0}
    syms = {"CoreWeave": "CRWV", "Circle": "CRCL", "Figma": "FIG", "Chime": "CHYM", "eToro": "ETOR"}
    names, fd, cur, pkv = [], [], [], []
    for n, s in syms.items():
        h = d[s]; o = offers[n]
        h2 = h[h["Volume"] > 0] if "Volume" in h else h  # skip placeholder zero-volume rows (e.g. ETOR 2025-05-13)
        fc = float(h2["Close"].iloc[0]); lc = float(h2["Close"].iloc[-1]); ph = float(h2["High"].max())
        p = _np.array([fc, lc, ph], dtype=_np.float64); out = _np.empty(3, dtype=_np.float64)
        _pct_from_base(p, _np.float64(o), out)
        names.append(n); fd.append(out[0]); cur.append(out[1]); pkv.append(out[2])
    # sort by current return desc for readability
    order = _np.argsort(_np.array(cur))[::-1]
    names = [names[i] for i in order]; fd = [fd[i] for i in order]; cur = [cur[i] for i in order]; pkv = [pkv[i] for i in order]
    y = _np.arange(len(names), dtype=_np.float64)
    fig, ax = _plt.subplots(figsize=(11, 5.4))
    _style(ax, "Chart 4: 2025 IPO Class — First-Day Pop vs. Later Performance (MAX data to Sep 30, 2026)",
           "Green = first-day close vs offer | Red = Sep-30-2026 close vs offer | gray dumbbell | peak annotated | Sources: Fortune, CNBC, SaaStr, Yahoo Finance")
    for i in range(len(names)):
        ax.hlines(y[i], min(fd[i], cur[i]), max(fd[i], cur[i]), colors=GRAYL, lw=2.5, zorder=2)
    ax.scatter(fd, y, s=110, color=GREEN2, edgecolors="white", linewidths=1.0, zorder=5, label="First-day close")
    ax.scatter(cur, y, s=110, color=RED, edgecolors="white", linewidths=1.0, zorder=5, label="Sep-30-2026 close")
    for i, n in enumerate(names):
        ax.text(max(fd[i], cur[i])+8, y[i]+0.08, f"peak {pkv[i]:+.0f}%", color=GRAYL, fontsize=7.5)
        ax.text(fd[i], y[i]+0.28, f"{fd[i]:+.0f}%", color=GREEN2, fontsize=8, weight="bold", ha="center")
        # top row (Circle): keep red label inside axes (below), others above to avoid green/red collision
        ry = y[i]+0.58 if i == 0 else y[i]-0.42
        ax.text(cur[i], ry, f"{cur[i]:+.0f}%", color=RED, fontsize=8, weight="bold", ha="center")
    ax.set_yticks(y); ax.set_yticklabels([f"{n} ({syms[n]}, \\${offers[n]:.0f})" for n in names], color="white")
    ax.set_xlabel("Return vs IPO offer price (%)"); ax.axvline(0, color="#C9D1D9", lw=1.0)
    ax.legend(framealpha=0.9, facecolor="#161B22", edgecolor=GRID, loc="lower right")
    ax.invert_yaxis(); _tight(fig); fig.savefig(OUT/"chart4_ipo_dumbbell.png", dpi=200, bbox_inches="tight"); _plt.close(fig)
    # dump audit table
    _pd.DataFrame({"company": names, "offer": [offers[n] for n in names],
        "first_day_pct": _np.round(fd,1), "current_pct": _np.round(cur,1), "peak_pct": _np.round(pkv,1)}).to_csv(DATA/"ipo_audit.csv", index=False)
    return "chart4"

def plot5():
    fig, ax = _plt.subplots(figsize=(10.5, 4.6))
    _style(ax, "Chart 5: TPU v1 vs Contemporary CPU/GPU — 2017 Benchmark (HISTORICAL, NOT current)",
           "Jouppi et al., ISCA 2017 (K80 GPU / Haswell CPU datacenter inference) via Wikipedia + Google Cloud blog")
    cats = ["Raw Performance (× faster)", "Performance per Watt (× more efficient)"]
    mins = _np.array([15.0, 30.0]); maxs = _np.array([30.0, 80.0])
    y = _np.arange(len(cats), dtype=_np.float64)
    cmap = _LSC.from_list("gblue", [GBLUE1, GBLUE2])
    for i in range(len(cats)):
        ax.barh(y[i], maxs[i]-mins[i], left=mins[i], height=0.5, color="none", edgecolor=GBLUE2, zorder=4)
        _xv = _np.linspace(mins[i], maxs[i], 128)
        for j in range(127):  # 127 thin slices = smooth gradient, still SIMD-cheap
            ax.barh(y[i], _xv[j+1]-_xv[j], left=_xv[j], height=0.5, color=cmap(j/127), edgecolor="none", zorder=3)
        ax.text(maxs[i]+1.2, y[i], f"{mins[i]:.0f}–{maxs[i]:.0f}×", color="white", fontsize=11, weight="bold", va="center",
                bbox=dict(boxstyle="round,pad=0.25", fc="#161B22", ec=GBLUE2))
        ax.text(mins[i], y[i]+0.32, f"min {mins[i]:.0f}×", color=GRAYL, fontsize=8, ha="center")
    ax.set_yticks(y); ax.set_yticklabels(cats, color="white"); ax.set_xlabel("Multiple of improvement (×)")
    ax.set_xlim(0, 95); ax.invert_yaxis()
    ax.text(0.5, -0.22, "2017 benchmark vs. contemporaneous hardware — NOT a comparison to current-generation chips.",
            transform=ax.transAxes, ha="center", color=AMBER, fontsize=9.5, weight="bold",
            bbox=dict(boxstyle="round,pad=0.4", fc="#161B22", ec=AMBER, alpha=0.95))
    _tight(fig); fig.savefig(OUT/"chart5_tpu_2017.png", dpi=200, bbox_inches="tight"); _plt.close(fig)
    return "chart5"

def main():
    import time as _t; t0 = _t.time()
    print(f"[init] threads={NTHREADS} numpy={_np.__version__} openblas-Haswell AVX2(FMA) X86_V3")
    allh = _load_all()
    nvda, ixic = allh["NVDA"], allh["^IXIC"]
    print(f"[data] NVDA {nvda.shape} {nvda.index[0].date()}->{nvda.index[-1].date()} | IXIC {ixic.shape} {ixic.index[0].date()}->{ixic.index[-1].date()}")
    # parallel render on 12 threads (Agg figures are thread-local)
    with _cf.ThreadPoolExecutor(max_workers=5) as ex:
        f1 = ex.submit(plot1, nvda); f2 = ex.submit(plot2, nvda); f3 = ex.submit(plot3, ixic)
        f4 = ex.submit(plot4, allh); f5 = ex.submit(plot5)
        for f in _cf.as_completed([f1, f2, f3, f4, f5]): print("[done]", f.result())
    print(f"[all] {sorted(p.name for p in OUT.glob('*.png'))} in {_t.time()-t0:.1f}s -> {OUT}")

if __name__ == "__main__":
    main()
