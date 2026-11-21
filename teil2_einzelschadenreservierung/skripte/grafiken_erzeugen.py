import json, os, math
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); TEIL = os.path.dirname(HERE)
src = open(os.path.join(HERE, "sicr_nachbau_backtest.py")).read().split("# ------------------------------------------------------------------ Backtest")[0]
exec(compile(src, os.path.join(HERE, "sicr_nachbau_backtest.py"), "exec"))
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.ticker
from matplotlib.colors import LinearSegmentedColormap
OUT = os.path.join(TEIL, "grafiken"); os.makedirs(OUT, exist_ok=True)
NAVY, TEAL, ORANGE, GREY, RED, GOLD, LIGHT, BLUE = "#0B2545", "#13807A", "#E07A1F", "#8A94A6", "#B23A48", "#C9A227", "#DDE3EA", "#6C8EBF"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.titlesize": 13, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#E6E9EF", "axes.axisbelow": True,
    "svg.fonttype": "none", "legend.frameon": False})
SEQ = LinearSegmentedColormap.from_list("seq", ["#F4F7FA", "#9CC5C2", TEAL, NAVY])
FOOT = "Daten: Beispieldaten des R-Pakets sicr (D. Skowasch), nur Schadenzahlungen ohne Renten · Nachbau und Rechnung: Opus 5.5"
def save(fig, name, note=""):
    fig.text(0.01, -0.03, FOOT + (" · " + note if note else ""), fontsize=8, color="#6B7280")
    fig.savefig(f"{OUT}/{name}.svg", bbox_inches="tight"); plt.close(fig)
M = lambda v: np.asarray(v) / 1e6
b = pd.read_csv(os.path.join(TEIL, "ergebnisse", "backtest.csv")); full = json.load(open(os.path.join(TEIL, "ergebnisse", "full2023.json")))
METH = [("sicr", "Einzelschadenverfahren (sicr)", TEAL, "o"), ("cl_paid", "Chain-Ladder Zahlungen", NAVY, "s"),
        ("cl_inc", "Chain-Ladder Schadenaufwand", BLUE, "^"), ("case", "Einzelfallreserven", GOLD, "D"), ("boot_mean", "ODP-Bootstrap (Mittel)", GREY, "v")]

# 1 Mechanik: simulierte Pfade eines Beispielschadens
K = 2023; x = prepare(K); pools = build_pools(x, K)
pay, cout, start, length = pools
last = x[x.Calendar_year == K].set_index("Claim_id"); last = last[last.ls <= K]
cand = last[(last.c_out == 3) & ((K - last.ls + 1) == 4)]
cid = cand.index[0]
hist = x[x.Claim_id == cid]
rng = np.random.default_rng(5)
fig, axs = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [1.15, 1]})
hd = hist[hist.dev >= 1]
axs[0].step(hd.Calendar_year, hd.c_out, where="post", color=NAVY, lw=3, label="beobachtet")
paths_pay = []
for s in range(25):
    c = int(last.loc[cid, "c_out"]); d0 = int(K - last.loc[cid, "ls"] + 1); cs = [c]; pp = []
    for y in range(1, 16):
        d = d0 + y; dd = min(d, SOT); r = start[dd, c] + int(rng.random() * length[dd, c])
        pp.append(pay[r] * tail_factor(d)); c = int(cout[r]); cs.append(c)
    paths_pay.append(np.cumsum(pp))
    axs[0].step(np.arange(K, K + 16), np.array(cs) + rng.normal(0, 0.05), where="post", color=TEAL, alpha=0.35, lw=1.2)
axs[0].axvline(K + 0.5, color=GREY, ls="--"); axs[0].text(K + 0.7, 5.2, "Stichtag", color=GREY, fontsize=9)
axs[0].set_yticks(range(6)); axs[0].set_yticklabels(["0: geschlossen", "1: < 200 T€", "2: 200–400 T€", "3: 400–700 T€", "4: 0,7–1,4 Mio. €", "5: > 1,4 Mio. €"], fontsize=9)
axs[0].set_title("Reserveklasse: 25 simulierte Zukünfte"); axs[0].set_xlabel("Kalenderjahr"); axs[0].xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
pp = np.array(paths_pay)
for q, a in [((5, 95), 0.2), ((25, 75), 0.4)]:
    axs[1].fill_between(np.arange(1, 16), np.percentile(pp, q[0], 0) / 1000, np.percentile(pp, q[1], 0) / 1000, color=TEAL, alpha=a, lw=0)
axs[1].plot(np.arange(1, 16), np.median(pp, 0) / 1000, color=NAVY, lw=2.2, label="Median")
axs[1].axhline(last.loc[cid, "ires"] / 1000, color=ORANGE, ls="--", label="heutige Einzelfallreserve")
axs[1].set_xlabel("Jahre nach Stichtag"); axs[1].set_ylabel("kumulierte künftige Zahlung (T€, indexiert)"); axs[1].legend(); axs[1].set_title("Zahlungsband desselben Schadens")
save(fig, "m01_mechanik", f"Schaden {cid}, groß seit {int(last.loc[cid,'ls'])}")

# 2 Poolgrößen
pl = np.array(full["pool_len"])[2:, :]
fig, ax = plt.subplots(figsize=(12, 3.8))
im = ax.imshow(pl.T, cmap=SEQ, aspect="auto")
for i in range(pl.shape[0]):
    for j in range(pl.shape[1]):
        ax.text(i, j, pl[i, j], ha="center", va="center", fontsize=8, color="white" if pl[i, j] > pl.max() * 0.5 else NAVY)
ax.set_xticks(range(pl.shape[0])); ax.set_xticklabels([str(d) for d in range(2, SOT)] + ["17+"])
ax.set_yticks(range(6)); ax.set_yticklabels(["0", "1", "2", "3", "4", "5"]); ax.set_xlabel("Jahr seit Großwerden"); ax.set_ylabel("Reserveklasse"); ax.grid(False)
ax.set_title("Wie viele historische Beobachtungen stehen je Zustand zur Verfügung? (Pools zum Stichtag 2023)")
save(fig, "m02_poolgroessen", "1 = leerer Pool, ersetzt durch Platzhalter ohne Zahlung")

# 3 Übergangsmatrix
p = x[(x.Calendar_year > K - 10) & (x.dev >= 2) & (x.dev <= 6)]
T = pd.crosstab(p.c_in, p.c_out, normalize="index").reindex(index=range(6), columns=range(6), fill_value=0)
fig, ax = plt.subplots(figsize=(6.6, 5))
ax.imshow(T.values, cmap=SEQ, vmin=0, vmax=1)
for i in range(6):
    for j in range(6): ax.text(j, i, f"{T.values[i,j]*100:.0f} %", ha="center", va="center", fontsize=9, color="white" if T.values[i, j] > .5 else NAVY)
ax.set_xticks(range(6)); ax.set_yticks(range(6)); ax.set_xlabel("Reserveklasse am Jahresende"); ax.set_ylabel("Reserveklasse am Jahresanfang"); ax.grid(False)
ax.set_title("Übergangshäufigkeiten (Jahre 2–6 seit Großwerden)")
save(fig, "m03_uebergang")

# 4 Backtest Prognose vs. Ist
fig, ax = plt.subplots(figsize=(12, 4.6))
ax.fill_between(b.K, M(b.sicr_q05), M(b.sicr_q95), color=TEAL, alpha=0.15, label="sicr 90 %-Band")
for col, lab, c, mk in METH:
    ax.plot(b.K, M(b[col]), marker=mk, color=c, lw=1.8 if col == "sicr" else 1.2, label=lab, alpha=1 if col in ("sicr", "cl_paid") else 0.8)
ax.plot(b.K, M(b.actual), "k-", lw=3, marker="o", label="tatsächlich gezahlt")
ax.set_xticks(b.K); ax.set_xlabel("Stichtag (Ende Kalenderjahr)"); ax.set_ylabel("Zahlungen der folgenden 5 Jahre (Mio. €)")
ax.legend(ncol=3, fontsize=9, loc="upper left"); ax.set_ylim(0, 65)
ax.set_title("Rollierender Backtest: Prognose der nächsten 5 Jahre gegen tatsächliche Zahlungen")
save(fig, "m04_backtest", "nur zum Stichtag bekannte Großschäden; Inflation = Mittel der letzten 10 Jahre")

# 5 Fehlerkennzahlen
err = {lab: (b[col] / b.actual - 1) * 100 for col, lab, c, mk in METH}
fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.2))
labs = [m[1] for m in METH]; cols = [m[2] for m in METH]
bias = [err[l].mean() for l in labs]; mape = [err[l].abs().mean() for l in labs]
axs[0].barh(labs[::-1], mape[::-1], color=cols[::-1])
for i, v in enumerate(mape[::-1]): axs[0].text(v + 1, i, f"{v:.0f} %", va="center", fontsize=10)
axs[0].set_xlabel("mittlerer absoluter Fehler in %"); axs[0].set_title("Treffgenauigkeit über 11 Stichtage")
bp = axs[1].boxplot([err[l] for l in labs], vert=True, patch_artist=True, widths=0.55)
for patch, c in zip(bp["boxes"], cols): patch.set_facecolor(c); patch.set_alpha(0.75)
for med in bp["medians"]: med.set_color("white"); med.set_linewidth(2)
axs[1].axhline(0, color="black", lw=1); axs[1].set_xticks(range(1, 6)); axs[1].set_xticklabels(["sicr", "CL Zahl.", "CL Aufw.", "Einzelres.", "Bootstrap"], fontsize=9)
axs[1].set_ylabel("Abweichung Prognose zu Ist in %"); axs[1].set_title("Streuung der Abweichung je Verfahren")
save(fig, "m05_fehler")

# 6 Kalibrierung der Intervalle
fig, ax = plt.subplots(figsize=(12, 4))
xk = b.K.values
ax.vlines(xk - 0.15, M(b.sicr_q05), M(b.sicr_q95), color=TEAL, lw=7, alpha=0.7, label="sicr 90 %-Intervall")
ax.vlines(xk + 0.15, M(b.boot_q05), M(b.boot_q95), color=GREY, lw=7, alpha=0.7, label="ODP-Bootstrap 90 %-Intervall")
ax.plot(xk, M(b.actual), "ko", ms=8, label="tatsächlich")
inn = ((b.actual >= b.sicr_q05) & (b.actual <= b.sicr_q95)).sum(); innb = ((b.actual >= b.boot_q05) & (b.actual <= b.boot_q95)).sum()
ax.set_xticks(xk); ax.set_ylabel("Mio. €"); ax.legend(ncol=3, fontsize=9)
ax.set_title(f"Intervalle zu schmal: Ist liegt bei sicr in {inn} von 11, beim Bootstrap in {innb} von 11 Fällen im 90 %-Band")
save(fig, "m06_kalibrierung")

# 7 Bewertung 2023
fig, axs = plt.subplots(1, 2, figsize=(13, 4.3), gridspec_kw={"width_ratios": [1, 1.1]})
vals = [full["sicr_mean"], full["cl_paid_all"], full["cl_inc_res"], full["case_res"]]
labs7 = ["sicr\n(Best Estimate)", "Chain-Ladder\nZahlungen", "Chain-Ladder\nSchadenaufwand", "Einzelfall-\nreserven (HGB)"]
axs[0].bar(labs7, M(vals), color=[TEAL, NAVY, BLUE, GOLD])
for i, v in enumerate(vals): axs[0].text(i, v / 1e6 + 2, f"{v/1e6:.1f}", ha="center", fontsize=10)
axs[0].set_ylabel("künftige Zahlungen, nominal (Mio. €)"); axs[0].set_title(f"Reserve {full['n_claims']} Großschäden, 2023")
dist = M(full["dist"])
axs[1].hist(dist, bins=40, color=LIGHT, edgecolor="white", density=True)
for q, lab, c in [(full["sicr_q"][1], "Median", TEAL), (full["sicr_q"][4], "99,5 %", RED)]:
    axs[1].axvline(q / 1e6, color=c, ls="--"); axs[1].text(q / 1e6, axs[1].get_ylim()[1] * 0.85, f" {lab}\n {q/1e6:.1f}", color=c, fontsize=9)
axs[1].set_yticks([]); axs[1].set_xlabel("Mio. €"); axs[1].set_title("sicr: 1.000 Simulationen")
save(fig, "m07_bewertung2023", "Inflation künftig 3 % p. a.; ohne Renten, IBNR und Sonderschäden")

# 8 Zahlungsmuster 2023
fig, ax = plt.subplots(figsize=(12, 3.8))
yy = np.arange(2024, 2054)
ax.bar(yy - 0.2, M(full["cf_sicr"]), 0.4, color=TEAL, label="sicr")
ax.bar(yy + 0.2, M(full["cf_clp"]), 0.4, color=NAVY, label="Chain-Ladder Zahlungen")
ax.set_ylabel("Mio. € je Jahr"); ax.legend(); ax.set_title("Auszahlungsprofil: sicr verteilt die Zahlungen flacher und länger")
save(fig, "m08_cashflow")

# 9 Sensitivität
s = full["sens"]; base = s["Basis"]["mean"]
items = [(k, v["mean"]) for k, v in s.items() if k != "Basis"]
fig, ax = plt.subplots(figsize=(11, 3.8))
for i, (k, v) in enumerate(items):
    d = (v / base - 1) * 100; ax.barh(i, d, color=TEAL if d > 0 else RED); ax.text(d + (0.5 if d >= 0 else -0.5), i, f"{d:+.1f} %  ({s[k]['n']} Schäden)", va="center", ha="left" if d >= 0 else "right", fontsize=9)
ax.set_yticks(range(len(items))); ax.set_yticklabels([k for k, _ in items]); ax.axvline(0, color="black")
ax.set_xlabel("Veränderung des Best Estimate ggü. Basis in %"); ax.set_xlim(-30, 30)
ax.set_title("Sensitivität: Schwelle und Poolfenster wirken stärker als die Klassengrenzen")
save(fig, "m09_sensitivitaet", "Schwellenvarianten ändern auch den betrachteten Bestand")

# 10 XL
fig, ax = plt.subplots(figsize=(12, 4))
ax.vlines(xk, b.xl_sicr_q05 / 1e6, b.xl_sicr_q95 / 1e6, color=TEAL, lw=7, alpha=0.35, label="sicr 90 %-Intervall")
ax.plot(xk, M(b.xl_sicr), "o-", color=TEAL, label="sicr Erwartungswert")
ax.plot(xk, M(b.xl_gn), "s-", color=NAVY, label="Brutto-Netto-Quote × CL")
ax.plot(xk, M(b.xl_actual), "k-", marker="o", lw=2.5, label="tatsächlich zediert")
ax.set_xticks(xk); ax.set_ylabel("XL-Anteil der nächsten 5 Jahre (Mio. €)"); ax.legend(ncol=2, fontsize=9)
ax.set_title("Rückversicherung: bis 2014 treffen beide, ab 2015 verpassen beide den Anstieg")
save(fig, "m10_xl", "XL je Schaden nach Quote, ohne Indexklausel (vereinfachte Annahme)")
print("ok")
