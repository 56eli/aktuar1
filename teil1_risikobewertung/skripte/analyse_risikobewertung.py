"""Aktuarielle Analyse des synthetischen Testsets aus github.com/56eli/aktuartestrepo und Erzeugung aller Grafiken.

Aufruf (aus dem Ordner teil1_risikobewertung):
    python3 skripte/analyse_risikobewertung.py [DATENSATZ_ORDNER] [AUSGABE_ORDNER]
Standard: daten/aktuartestrepo/data/synthetic/aktuariat-risikobewertung-v1 und der Ordner teil1_risikobewertung.
"""
import sys, json, os, math
import numpy as np, pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon
from matplotlib.colors import LinearSegmentedColormap

TEIL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DS = sys.argv[1] if len(sys.argv) > 1 else os.path.join(TEIL, "daten", "aktuartestrepo", "data", "synthetic", "aktuariat-risikobewertung-v1")
OUT = sys.argv[2] if len(sys.argv) > 2 else TEIL
CH = os.path.join(OUT, "grafiken"); os.makedirs(CH, exist_ok=True)
ERG = os.path.join(OUT, "ergebnisse"); os.makedirs(ERG, exist_ok=True)
rng = np.random.default_rng(20261121)
R = {}  # Ergebnisse

# ---------- Stil ----------
NAVY, TEAL, ORANGE, GREY, RED, GOLD, LIGHT = "#0B2545", "#13807A", "#E07A1F", "#8A94A6", "#B23A48", "#C9A227", "#DDE3EA"
PAL = [NAVY, TEAL, ORANGE, "#6C8EBF", RED, GOLD, "#5B5F97"]
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#5A6475", "axes.labelcolor": "#2B3445", "xtick.color": "#2B3445", "ytick.color": "#2B3445",
    "axes.grid": True, "grid.color": "#E6E9EF", "grid.linewidth": 0.8, "axes.axisbelow": True,
    "svg.fonttype": "none", "figure.dpi": 110, "legend.frameon": False,
})
SEQ = LinearSegmentedColormap.from_list("seq", ["#F4F7FA", "#9CC5C2", TEAL, NAVY])
DIV = LinearSegmentedColormap.from_list("div", [RED, "#F2D0D4", "#F7F7F7", "#BFE0DD", TEAL])
FOOT = "Testdaten / synthetisch – kein tatsächlicher Solvenz- oder Unternehmensbefund · Stichtag 31.12.2025"
CHARTS = []

def eur(x, d=0):
    s = f"{x:,.{d}f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return s
def teur(x): return eur(x / 1000, 0)

def save(fig, name, title, kind, note=""):
    fig.text(0.01, -0.02, FOOT + (" · " + note if note else ""), fontsize=7, color="#6B7280", ha="left", va="bottom")
    fig.savefig(os.path.join(CH, name + ".svg"), bbox_inches="tight")
    plt.close(fig)
    CHARTS.append({"file": name + ".svg", "title": title, "kind": kind})

# ---------- Daten ----------
p = pd.read_csv(f"{DS}/clean/policy_exposure.csv")
c = pd.read_csv(f"{DS}/clean/claim_header.csv", parse_dates=["loss_date", "report_date"])
d = pd.read_csv(f"{DS}/clean/claim_development.csv", parse_dates=["valuation_date"])
xl = pd.read_csv(f"{DS}/clean/reinsurance_treaties.csv")
cat = pd.read_csv(f"{DS}/clean/catastrophe_events.csv")
mc = pd.read_csv(f"{DS}/clean/market_curve.csv")
sc = pd.read_csv(f"{DS}/clean/scenario_catalog.csv")
truth = pd.read_csv(f"{DS}/reference/claim_ultimate_truth.csv")  # nur zur Nachprüfung (Backtest)
CUT = pd.Timestamp("2025-12-31")

# letzter Stand je Schaden zum Stichtag
last = d.sort_values("valuation_date").groupby("claim_id").tail(1).set_index("claim_id")
c = c.set_index("claim_id").join(last[["accident_year", "gross_paid_cumulative_eur", "gross_incurred_eur", "ceded_incurred_eur", "net_incurred_eur"]])
c["delay_days"] = (c.report_date - c.loss_date).dt.days
R["n_policies"] = int(p.policy_id.nunique()); R["n_exposure_rows"] = len(p)
R["exposure_years"] = float(p.exposure_years.sum()); R["premium"] = float(p.gross_premium_eur.sum())
R["n_claims"] = len(c); R["incurred_cut"] = float(c.gross_incurred_eur.sum()); R["paid_cut"] = float(c.gross_paid_cumulative_eur.sum())
R["open_claims"] = int((c.claim_status_at_cutoff == "open").sum())

# ---------- 1 Portfolio ----------
cnt = c.groupby("exposure_id").size()
p["n"] = p.exposure_id.map(cnt).fillna(0)
inc = c.groupby("exposure_id").gross_incurred_eur.sum()
p["inc"] = p.exposure_id.map(inc).fillna(0)
py = p.groupby(["policy_year", "product"]).agg(expo=("exposure_years", "sum"), prem=("gross_premium_eur", "sum"), n=("n", "sum"), inc=("inc", "sum")).reset_index()

fig, axs = plt.subplots(1, 2, figsize=(11, 3.8))
for ax, col, lab in [(axs[0], "expo", "Policenjahre (Exposition)"), (axs[1], "prem", "Bruttoprämie in T€")]:
    bottom = np.zeros(6)
    for i, prod in enumerate(["motor", "property"]):
        v = py[py["product"] == prod][col].values / (1000 if col == "prem" else 1)
        ax.bar(py.policy_year.unique(), v, bottom=bottom, color=PAL[i], label={"motor": "Kraftfahrt", "property": "Sach"}[prod], width=0.65)
        bottom += v
    ax.set_title(lab); ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2)
fig.suptitle("Bestand stabil: rund 1.680 Policenjahre und 580 T€ Prämie je Jahr", x=0.01, ha="left", fontweight="bold")
save(fig, "01_bestand", "Bestand und Prämie je Policenjahr", "Gestapeltes Säulendiagramm")

# Frequenz nach Merkmal mit Poisson-KI
def freq_tab(col):
    g = p.groupby(col).agg(n=("n", "sum"), e=("exposure_years", "sum"))
    g["f"] = g.n / g.e
    g["lo"] = stats.chi2.ppf(0.025, 2 * g.n) / 2 / g.e
    g["hi"] = stats.chi2.ppf(0.975, 2 * (g.n + 1)) / 2 / g.e
    return g
LBL = {"region": "Region", "risk_band": "Risikoklasse", "driver_age_band": "Fahreralter", "asset_age_band": "Fahrzeug-/Objektalter"}
fig, axs = plt.subplots(1, 4, figsize=(13, 3.8), sharey=True)
ft = {}
for ax, col in zip(axs, LBL):
    g = freq_tab(col); ft[col] = g
    x = np.arange(len(g))
    ax.errorbar(x, g.f * 100, yerr=[(g.f - g.lo) * 100, (g.hi - g.f) * 100], fmt="o", color=NAVY, ecolor=TEAL, capsize=4, lw=1.6)
    ax.set_xticks(x); ax.set_xticklabels([str(i).replace("_", "–") for i in g.index], rotation=30, fontsize=8)
    for xi, (n, f) in enumerate(zip(g.n, g.f)): ax.annotate(f"n={int(n)}", (xi, g.hi.iloc[xi] * 100), textcoords="offset points", xytext=(0, 4), ha="center", fontsize=7, color=GREY)
    ax.set_title(LBL[col], fontsize=10)
axs[0].set_ylabel("Schäden je 100 Policenjahre")
fig.suptitle("Risikoklasse ist der klarste Frequenztreiber (95 %-Poisson-Konfidenzintervalle)", x=0.01, ha="left", fontweight="bold")
save(fig, "02_frequenz_merkmale", "Schadenfrequenz je Tarifmerkmal mit Konfidenzintervall", "Punktdiagramm mit Fehlerbalken")
R["freq_total"] = float(p.n.sum() / p.exposure_years.sum())
R["freq_by_band"] = {int(k): round(float(v) * 100, 2) for k, v in ft["risk_band"].f.items()}

# Zweidimensionale Frequenz-Heatmap
h = p.groupby(["region", "risk_band"]).agg(n=("n", "sum"), e=("exposure_years", "sum"))
hm = (h.n / h.e * 100).unstack(); he = h.e.unstack()
fig, ax = plt.subplots(figsize=(7.5, 4))
im = ax.imshow(hm.values, cmap=SEQ, aspect="auto")
for i in range(hm.shape[0]):
    for j in range(hm.shape[1]):
        v = hm.values[i, j]; ax.text(j, i, f"{v:.1f}\n({he.values[i,j]:.0f} PJ)", ha="center", va="center", fontsize=7.5, color="white" if v > np.nanpercentile(hm.values, 60) else NAVY)
ax.set_xticks(range(5)); ax.set_xticklabels([f"Klasse {k}" for k in hm.columns]); ax.set_yticks(range(5)); ax.set_yticklabels(hm.index.str.capitalize())
ax.grid(False); fig.colorbar(im, ax=ax, label="Schäden je 100 PJ")
ax.set_title("Frequenz-Heatmap Region × Risikoklasse (mit Exposition)")
save(fig, "03_heatmap_region_klasse", "Frequenz-Heatmap Region × Risikoklasse", "Heatmap mit Werten")

# Schwere nach Schadenart (abgeschlossene Schäden)
cl = c[c.claim_status_at_cutoff == "closed"]
order = cl.groupby("claim_type").gross_incurred_eur.median().sort_values().index
DE = {"glass": "Glas", "collision": "Kollision", "liability": "Haftpflicht", "water": "Leitungswasser", "storm": "Sturm", "other": "Sonstige", "fire": "Feuer", "flood": "Überschwemmung", "hail": "Hagel"}
fig, ax = plt.subplots(figsize=(10, 4.2))
data = [cl[cl.claim_type == t].gross_incurred_eur.values for t in order]
vp = ax.violinplot(data, showextrema=False, widths=0.8)
for b in vp["bodies"]: b.set_facecolor(LIGHT); b.set_alpha(1); b.set_edgecolor("none")
ax.boxplot(data, widths=0.25, patch_artist=True, boxprops=dict(facecolor=TEAL, color=NAVY), medianprops=dict(color="white", lw=2), flierprops=dict(marker=".", markersize=4, markeredgecolor=GREY))
ax.set_yscale("log"); ax.set_xticks(range(1, len(order) + 1)); ax.set_xticklabels([f"{DE[t]}\n(n={len(x)})" for t, x in zip(order, data)], fontsize=8)
ax.set_ylabel("Schadenaufwand in € (log)"); ax.set_title("Schadenschwere: Elementarschäden liegen eine Größenordnung über Glas")
save(fig, "04_schwere_violin", "Schadenschwere je Schadenart", "Box-/Violinplot (log)", "nur abgeschlossene Schäden")

# Meldeverzug
fig, axs = plt.subplots(1, 2, figsize=(11, 3.6))
axs[0].hist(c.delay_days, bins=30, color=TEAL, edgecolor="white"); axs[0].set_xlabel("Tage zwischen Schadeneintritt und Meldung"); axs[0].set_ylabel("Anzahl Schäden"); axs[0].set_title("Histogramm Meldeverzug")
xs = np.sort(c.delay_days); axs[1].step(xs, np.arange(1, len(xs) + 1) / len(xs) * 100, color=NAVY, lw=2)
for q in [0.5, 0.9, 0.99]:
    v = np.quantile(xs, q); axs[1].axvline(v, color=ORANGE, ls="--", lw=1); axs[1].text(v + 3, q * 100 - 8, f"{int(q*100)} %: {v:.0f} T.", fontsize=8, color=ORANGE)
axs[1].set_title("Empirische Verteilungsfunktion"); axs[1].set_ylabel("Anteil gemeldet in %"); axs[1].set_xlabel("Tage")
save(fig, "05_meldeverzug", "Meldeverzug (Histogramm und ECDF)", "Histogramm + Verteilungsfunktion")
R["delay_q90"] = float(np.quantile(xs, 0.9))

# ---------- 2 GLM Frequenz mit Holdout ----------
feats = ["product", "region", "risk_band", "driver_age_band", "asset_age_band"]
X = pd.get_dummies(p[feats].astype(str), drop_first=False)
base = {"product_motor", "region_central", "risk_band_1", "driver_age_band_25_39", "asset_age_band_0_5"}
# not_applicable driver age is collinear with property -> drop
drop = list(base) + ["driver_age_band_not_applicable"]
X = X.drop(columns=[k for k in drop if k in X]).astype(float)
X.insert(0, "const", 1.0)
y, off = p.n.values, np.log(p.exposure_years.values)
train = p.policy_year <= 2022; test = ~train
def poisson_irls(Xm, y, off, it=30):
    b = np.zeros(Xm.shape[1]); b[0] = np.log(y.sum() / np.exp(off).sum())
    for _ in range(it):
        eta = Xm @ b + off; mu = np.exp(eta); W = mu
        z = eta - off + (y - mu) / mu
        XtW = Xm.T * W
        b_new = np.linalg.solve(XtW @ Xm + 1e-8 * np.eye(len(b)), XtW @ z)
        if np.max(abs(b_new - b)) < 1e-9: b = b_new; break
        b = b_new
    cov = np.linalg.inv((Xm.T * np.exp(Xm @ b + off)) @ Xm)
    return b, np.sqrt(np.diag(cov))
Xm = X.values
b, se = poisson_irls(Xm[train.values], y[train.values], off[train.values])
mu_all = np.exp(Xm @ b + off); p["mu"] = mu_all
coef = pd.DataFrame({"b": b, "se": se}, index=X.columns)
R["glm_coef"] = {k: [round(float(np.exp(v.b)), 3), round(float(np.exp(v.b - 1.96 * v.se)), 3), round(float(np.exp(v.b + 1.96 * v.se)), 3)] for k, v in coef.iterrows()}
tst = p[test.values]
R["glm_holdout_AE"] = float(tst.n.sum() / tst.mu.sum())
# Poisson-Deviance Holdout vs Nullmodell
def dev(yv, m): return 2 * np.sum(np.where(yv > 0, yv * np.log(np.where(yv > 0, yv, 1) / m), 0) - (yv - m))
null_mu = tst.exposure_years * (p[train.values].n.sum() / p[train.values].exposure_years.sum())
R["glm_dev_model"] = float(dev(tst.n.values, tst.mu.values)); R["glm_dev_null"] = float(dev(tst.n.values, null_mu.values))

# Relativitäten-Plot
cf = coef.drop("const")
fig, ax = plt.subplots(figsize=(10, 4.6))
yy = np.arange(len(cf))[::-1]
rel, lo, hi = np.exp(cf.b), np.exp(cf.b - 1.96 * cf.se), np.exp(cf.b + 1.96 * cf.se)
ax.hlines(yy, lo, hi, color=TEAL, lw=2.5); ax.plot(rel, yy, "o", color=NAVY)
ax.axvline(1, color=GREY, ls="--"); ax.set_xscale("log")
nice = [k.replace("product_property", "Sparte Sach").replace("region_", "Region ").replace("risk_band_", "Risikoklasse ").replace("driver_age_band_", "Fahreralter ").replace("asset_age_band_", "Fahrzeugalter ").replace("_", "–") for k in cf.index]
ax.set_yticks(yy); ax.set_yticklabels(nice, fontsize=8); ax.set_xlabel("Relativität zur Basisklasse (log), 95 %-KI")
ax.set_title("Poisson-GLM: Risikoklasse 5 hat fast die doppelte Basisfrequenz")
save(fig, "06_glm_relativitaeten", "GLM-Relativitäten mit Konfidenzintervallen", "Faktor-/Forest-Plot", "Basis: Kraftfahrt, Region central, Klasse 1, 25–39 J., 0–5 J.; Training 2019–2022")

# Lift-Chart / Kalibrierung auf Holdout
t = tst.copy(); t["rate"] = t.mu / t.exposure_years
t["dec"] = pd.qcut(t.rate.rank(method="first"), 10, labels=False)
g = t.groupby("dec").agg(obs=("n", "sum"), pred=("mu", "sum"), e=("exposure_years", "sum"))
fig, axs = plt.subplots(1, 2, figsize=(11.5, 4))
x = np.arange(1, 11)
axs[0].bar(x, g.obs / g.e * 100, color=LIGHT, edgecolor=GREY, label="beobachtet")
axs[0].plot(x, g.pred / g.e * 100, "o-", color=NAVY, lw=2, label="prognostiziert")
axs[0].set_xlabel("Dezil der prognostizierten Frequenz (Holdout 2023–2024)"); axs[0].set_ylabel("Schäden je 100 PJ"); axs[0].legend(); axs[0].set_title("Lift-Chart")
axs[0].set_xticks(x)
# Kalibrierung
axs[1].scatter(g.pred / g.e * 100, g.obs / g.e * 100, s=g.e / 4, color=TEAL, alpha=0.8, edgecolor=NAVY)
mx = max((g.obs / g.e).max(), (g.pred / g.e).max()) * 110
axs[1].plot([0, mx], [0, mx], "--", color=GREY); axs[1].set_xlim(0, mx); axs[1].set_ylim(0, mx)
axs[1].set_xlabel("prognostiziert je 100 PJ"); axs[1].set_ylabel("beobachtet je 100 PJ"); axs[1].set_title(f"Kalibrierung (A/E gesamt {R['glm_holdout_AE']:.2f})")
save(fig, "07_lift_kalibrierung", "Lift-Chart und Kalibrierungsplot (Holdout)", "Lift-Chart + Kalibrierungsplot")

# Lorenz / Gini
o = t.sort_values("rate")
ce = np.cumsum(o.exposure_years) / o.exposure_years.sum(); cn = np.cumsum(o.n) / o.n.sum()
gini = 1 - 2 * np.trapezoid(np.r_[0, cn], np.r_[0, ce])
R["gini"] = float(gini)
fig, ax = plt.subplots(figsize=(5.2, 4.6))
ax.plot(np.r_[0, ce], np.r_[0, cn], color=NAVY, lw=2.2, label=f"GLM (Gini {gini:.2f})")
ax.plot([0, 1], [0, 1], "--", color=GREY, label="ohne Differenzierung")
ax.fill_between(np.r_[0, ce], np.r_[0, cn], np.r_[0, ce], color=TEAL, alpha=0.15)
ax.set_xlabel("kumulierter Anteil Exposition (nach Risiko sortiert)"); ax.set_ylabel("kumulierter Anteil Schäden"); ax.legend(loc="upper left")
ax.set_title("Lorenzkurve (Holdout)")
save(fig, "08_lorenz", "Lorenzkurve und Gini-Koeffizient", "Lorenzkurve")

# ---------- 3 Reservierung: Dreiecke ----------
AYS = list(range(2019, 2025))
def triangle(col):
    T = pd.DataFrame(index=AYS, columns=range(1, 8), dtype=float)
    dd = d.sort_values("valuation_date")
    for ay in AYS:
        sub = dd[dd.accident_year == ay]
        for k in range(1, 8):
            cal_end = pd.Timestamp(f"{ay + k - 1}-12-31")
            if cal_end > CUT: continue
            s = sub[sub.valuation_date <= cal_end].groupby("claim_id").tail(1)
            T.loc[ay, k] = s[col].sum()
    return T
TI, TP = triangle("gross_incurred_eur"), triangle("gross_paid_cumulative_eur")
TN = triangle("net_incurred_eur")

def chain_ladder(T):
    n = T.shape[1]; f = []
    for k in range(1, n):
        m = T[k + 1].notna() & T[k].notna() & (T[k] > 0)
        f.append(T.loc[m, k + 1].sum() / T.loc[m, k].sum() if m.sum() else 1.0)
    f = np.array(f)
    ult, lastv, lastk = {}, {}, {}
    for ay in T.index:
        row = T.loc[ay].dropna(); k = row.index.max(); lastv[ay] = row.iloc[-1]; lastk[ay] = k
        ult[ay] = row.iloc[-1] * np.prod(f[k - 1:])
    return f, pd.Series(ult), pd.Series(lastv), pd.Series(lastk)
fI, uI, lI, kI = chain_ladder(TI); fP, uP, lP, kP = chain_ladder(TP)

def mack(T, f):
    n = T.shape[1]; sig2 = []
    for k in range(1, n):
        m = T[k + 1].notna() & (T[k] > 0)
        if m.sum() > 1:
            sig2.append(((T.loc[m, k] * (T.loc[m, k + 1] / T.loc[m, k] - f[k - 1]) ** 2).sum()) / (m.sum() - 1))
        else:
            sig2.append(min(sig2[-1] ** 2 / sig2[-2], min(sig2[-1], sig2[-2])) if len(sig2) >= 2 and sig2[-2] > 0 else (sig2[-1] if sig2 else 0))
    sig2 = np.array(sig2); res = {}
    for ay in T.index:
        row = T.loc[ay].dropna(); k0 = row.index.max(); C = row.iloc[-1]; U = C * np.prod(f[k0 - 1:])
        s = 0
        for k in range(k0, n):
            Ck = C * np.prod(f[k0 - 1:k - 1]) if k > k0 else C
            colsum = T.loc[T[k + 1].notna(), k].sum()
            s += sig2[k - 1] / f[k - 1] ** 2 * (1 / Ck + 1 / colsum)
        res[ay] = U * math.sqrt(s)
    return pd.Series(res), sig2
seI, sig2I = mack(TI, fI)

prem_ay = p.groupby("policy_year").gross_premium_eur.sum()
# BF: a-priori-Schadenquote aus reiferen Jahren 2019–2021 (CL incurred)
elr = float((uI.loc[2019:2021].sum()) / prem_ay.loc[2019:2021].sum())
cdf = pd.Series({ay: np.prod(fI[kI[ay] - 1:]) for ay in AYS})
bf = lI + prem_ay * elr * (1 - 1 / cdf)
tr = truth.set_index("claim_id").join(c[["accident_year"]]).groupby("accident_year")[["gross_ultimate_eur", "net_ultimate_eur", "ceded_ultimate_eur"]].sum()
resv = pd.DataFrame({"Prämie": prem_ay, "Incurred": lI, "Paid": lP, "CL_inc": uI, "CL_paid": uP, "BF": bf, "Truth": tr.gross_ultimate_eur, "Mack_SE": seI})
resv["Reserve_CL"] = resv.CL_inc - resv.Paid
resv["IBNR_CL"] = resv.CL_inc - resv.Incurred
R["elr_apriori"] = elr
R["f_inc"] = [round(float(v), 4) for v in fI]; R["f_paid"] = [round(float(v), 4) for v in fP]
mack_total = math.sqrt((seI ** 2).sum())  # vereinfachend ohne Kovarianzterm
R["reserve_table"] = {int(ay): {k: round(float(v), 2) for k, v in r.items()} for ay, r in resv.iterrows()}
R["tot"] = {k: float(resv[k].sum()) for k in ["Prämie", "Incurred", "Paid", "CL_inc", "CL_paid", "BF", "Truth", "Reserve_CL", "IBNR_CL"]}
R["mack_total_se_approx"] = mack_total
R["case_reserve_cut"] = R["incurred_cut"] - R["paid_cut"]

def tri_heat(T, name, title):
    U = T.copy(); f = fI if T is TI else fP
    fig, ax = plt.subplots(figsize=(10, 4.3))
    vals = T.values / 1000
    im = ax.imshow(np.where(np.isnan(vals), np.nan, vals), cmap=SEQ, aspect="auto")
    for i, ay in enumerate(AYS):
        row = T.loc[ay]; k0 = row.dropna().index.max(); C = row.dropna().iloc[-1]
        for j, k in enumerate(range(1, 8)):
            if not np.isnan(row[k]):
                ax.text(j, i, eur(row[k] / 1000), ha="center", va="center", fontsize=8, color="white" if row[k] / 1000 > np.nanmax(vals) * 0.55 else NAVY)
            else:
                proj = C * np.prod(f[k0 - 1:k - 1])
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor="#FFF6EC", edgecolor="white", hatch="///", lw=0))
                ax.text(j, i, eur(proj / 1000), ha="center", va="center", fontsize=8, color=ORANGE, style="italic")
    ax.set_xticks(range(7)); ax.set_xticklabels([f"{k}" for k in range(1, 8)]); ax.set_yticks(range(6)); ax.set_yticklabels(AYS)
    ax.set_xlabel("Abwicklungsjahr"); ax.set_ylabel("Anfalljahr"); ax.grid(False)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="T€ (beobachtet)")
    save(fig, name, title, "Abwicklungsdreieck als Heatmap", "orange kursiv = Chain-Ladder-Projektion")
tri_heat(TI, "09_dreieck_incurred", "Abwicklungsdreieck Schadenaufwand (Incurred, brutto, T€)")
tri_heat(TP, "10_dreieck_paid", "Abwicklungsdreieck Zahlungen (Paid, brutto, T€)")

# Link Ratios
fig, ax = plt.subplots(figsize=(10, 4))
for i, ay in enumerate(AYS):
    row = TI.loc[ay].dropna(); lr = (row.values[1:] / row.values[:-1])
    if len(lr): ax.plot(np.arange(1, len(lr) + 1), lr, "o-", color=PAL[i % 7], alpha=0.75, lw=1.2, ms=4, label=str(ay))
ax.plot(np.arange(1, 7), fI, "s-", color="black", lw=2.5, ms=6, label="volumengewichtet")
for k, v in enumerate(fI): ax.text(k + 1, v + 0.01, f"{v:.3f}", ha="center", fontsize=8)
ax.axhline(1, color=GREY, ls=":"); ax.set_xticks(range(1, 7)); ax.set_xticklabels([f"{k}→{k+1}" for k in range(1, 7)])
ax.set_ylabel("Abwicklungsfaktor (Incurred)"); ax.legend(ncol=7, fontsize=8, loc="upper right"); ax.set_title("Abwicklungsfaktoren je Anfalljahr: Incurred schwingt über 1,0 hinaus und kommt zurück")
save(fig, "11_linkratios", "Abwicklungsfaktoren je Anfalljahr", "Linienplot der Link Ratios")

# Abwicklungsmuster: Zahlungsanteil am Endschaden je Anfalljahr
fig, ax = plt.subplots(figsize=(9, 4))
for i, ay in enumerate(AYS):
    row = TP.loc[ay].dropna() / uI[ay] * 100
    ax.plot(row.index, row.values, "o-", color=PAL[i % 7], lw=1.6, ms=4, label=str(ay))
    ax.plot(row.index[-1], row.values[-1], "o", ms=9, mfc="none", mec=RED, mew=2)
ax.plot(range(1, 8), [np.nanmean((TI.loc[:, k] / uI * 100).values) for k in range(1, 8)], "k--", lw=2, label="Incurred (Mittel)")
ax.set_ylim(0, 110); ax.set_xlabel("Abwicklungsjahr"); ax.set_ylabel("% des Endschadens (CL Incurred)"); ax.legend(ncol=4, fontsize=8)
ax.set_title("Zahlungsmuster: jede letzte Diagonale (rot umrandet) springt nach oben")
save(fig, "12_muster", "Abwicklungsmuster Paid vs. Incurred", "Abwicklungsmuster-Kurven")
R["paid_pct_y1"] = float(np.nanmean((TP.loc[:2023, 1] / uI.loc[:2023]).values) * 100)

# Methodenvergleich
fig, ax = plt.subplots(figsize=(11, 4.2))
w = 0.18; x = np.arange(6)
for i, (col, lab, colr) in enumerate([("CL_paid", "Chain-Ladder Paid", "#6C8EBF"), ("CL_inc", "Chain-Ladder Incurred", NAVY), ("BF", "Bornhuetter-Ferguson", TEAL), ("Truth", "Referenzwert (Truth)", ORANGE)]):
    ax.bar(x + (i - 1.5) * w, resv[col] / 1000, w, color=colr, label=lab)
ax.errorbar(x - 0.5 * w, resv.CL_inc / 1000, yerr=1.96 * resv.Mack_SE / 1000, fmt="none", ecolor="black", capsize=3, lw=1)
ax.set_xticks(x); ax.set_xticklabels(AYS); ax.set_ylabel("Endschaden in T€"); ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.12))
ax.set_title("Methodenvergleich Endschaden je Anfalljahr (Fehlerbalken: ±1,96 Mack-SE)")
save(fig, "13_methodenvergleich", "Methodenvergleich CL / BF / Referenz", "Gruppiertes Säulendiagramm mit Fehlerbalken")

# Backtest A/E
fig, ax = plt.subplots(figsize=(9, 3.8))
for col, lab, colr, mk in [("CL_inc", "CL Incurred", NAVY, "o"), ("CL_paid", "CL Paid", "#6C8EBF", "^"), ("BF", "BF", TEAL, "s")]:
    ax.plot(AYS, (resv[col] / resv.Truth - 1) * 100, marker=mk, color=colr, lw=2, label=lab)
ax.axhline(0, color="black", lw=1); ax.axhspan(-5, 5, color=LIGHT, alpha=0.6, label="±5 %-Band")
ax.set_ylabel("Abweichung zur Referenz in %"); ax.legend(ncol=4); ax.set_title("Backtest gegen Referenzwerte: junge Anfalljahre streuen am stärksten")
save(fig, "14_backtest", "Backtest Schätzung vs. Referenz", "Abweichungsplot")

# Residuen (Paid) – Diagonaleffekt sichtbar machen
fig, axs = plt.subplots(1, 2, figsize=(11, 3.6))
rr = []
for ay in AYS:
    row = TP.loc[ay].dropna()
    for k in range(1, len(row)):
        rr.append((ay, k, ay + k, row.iloc[k] / row.iloc[k - 1] / fP[k - 1] - 1))
rr = pd.DataFrame(rr, columns=["ay", "k", "cal", "r"])
R["paid_resid_by_cal"] = {int(k): round(float(v) * 100, 1) for k, v in rr.groupby("cal").r.mean().items()}
for a, col, lab in [(axs[0], "k", "Abwicklungsjahr (Ausgang)"), (axs[1], "cal", "Kalenderjahr")]:
    colors = [RED if cy == 2025 else TEAL for cy in rr.cal]
    a.scatter(rr[col], rr.r * 100, c=colors, edgecolor=NAVY, s=45); a.axhline(0, color=GREY); a.set_xlabel(lab)
axs[0].set_ylabel("Abweichung Link Ratio vom Mittel in %"); axs[0].set_title("Residuen nach Abwicklungsjahr")
axs[1].set_title("Residuen nach Kalenderjahr: nur 2025 systematisch positiv")
save(fig, "15_residuen", "Residuendiagnostik Chain-Ladder (Paid)", "Residuenplots", "rot = Kalenderjahr 2025 (letzte Diagonale)")

# Reserveverteilung (Lognormal aus Mack) + Fan Chart Zahlungsstrom
res_mean = R["tot"]["CL_inc"] - R["tot"]["Paid"]
cv = mack_total / res_mean; s_ln = math.sqrt(math.log(1 + cv ** 2)); m_ln = math.log(res_mean) - s_ln ** 2 / 2
sim_res = rng.lognormal(m_ln, s_ln, 100000)
R["res_mean"] = res_mean; R["res_cv"] = cv; R["res_q995"] = float(np.quantile(sim_res, 0.995)); R["res_q75"] = float(np.quantile(sim_res, 0.75))
fig, ax = plt.subplots(figsize=(9, 3.8))
ax.hist(sim_res / 1000, bins=120, density=True, color=LIGHT, edgecolor="white")
xx = np.linspace(sim_res.min(), np.quantile(sim_res, 0.9995), 400)
ax.plot(xx / 1000, stats.lognorm.pdf(xx, s_ln, scale=math.exp(m_ln)) * 1000, color=NAVY, lw=2)
for q, lab, colr in [(0.5, "Median", TEAL), (0.75, "75 %", GOLD), (0.995, "99,5 %", RED)]:
    v = np.quantile(sim_res, q); ax.axvline(v / 1000, color=colr, ls="--"); ax.text(v / 1000, ax.get_ylim()[1] * 0.9, f" {lab}\n {eur(v/1000)} T€", color=colr, fontsize=8)
ax.set_xlabel("Rückstellung für noch nicht abgewickelte Schäden (brutto, T€)"); ax.set_yticks([]); ax.set_title(f"Reserveverteilung: Mittel {teur(res_mean)} T€, Variationskoeffizient {cv*100:.0f} %")
save(fig, "16_reserveverteilung", "Verteilung der Schadenreserve (Mack, lognormal)", "Dichte mit Quantilen")

# Zahlungsstrom-Projektion und Fan Chart (Paid-Faktoren ohne letzte Diagonale)
TPx = TP.copy()
for ay in AYS:
    k_last = TP.loc[ay].dropna().index.max(); TPx.loc[ay, k_last] = np.nan
fPx, _, _, _ = chain_ladder(TPx); fPx = np.where(np.isnan(fPx) | (fPx < 1), 1.0, fPx); R["f_paid_ex_diag"] = [round(float(v), 4) for v in fPx]
fut = {}  # erwartete künftige Zahlungen je Kalenderjahr
for ay in AYS:
    k0 = int(kP[ay]); C = lP[ay]; cum_prev = C
    for k in range(k0, 7):
        nxt = cum_prev * fPx[k - 1]; fut[ay + k] = fut.get(ay + k, 0) + (nxt - cum_prev); cum_prev = nxt
# Rest: Differenz CL incurred vs CL paid auf Jahre verteilen (Reserve gesamt auf Incurred-Basis)
cy = sorted(k for k in fut if fut[k] > 1); cf = np.array([fut[y] for y in cy]); cf = cf / cf.sum() * res_mean
R["cashflows"] = {int(y): float(v) for y, v in zip(cy, cf)}
paths = np.outer(sim_res / res_mean, cf) * rng.dirichlet(cf / cf.sum() * 200, 100000) / (cf / cf.sum())
cum_paid_hist = [float(d[d.valuation_date.dt.year == yr].groupby("claim_id").tail(1).gross_paid_cumulative_eur.sum()) for yr in range(2019, 2026)]
yrs = list(range(2026, 2026 + len(cy)))
fig, ax = plt.subplots(figsize=(10, 4))
qs = [(0.005, 0.995, 0.15), (0.05, 0.95, 0.25), (0.25, 0.75, 0.4)]
for lo_, hi_, a in qs:
    ax.fill_between(yrs, np.quantile(paths, lo_, axis=0) / 1000, np.quantile(paths, hi_, axis=0) / 1000, color=TEAL, alpha=a, lw=0, label=f"{int(lo_*1000)/10:g}–{int(hi_*1000)/10:g} %-Band")
ax.plot(yrs, np.median(paths, axis=0) / 1000, color=NAVY, lw=2.2, label="Median")
ax.set_xticks(yrs); ax.set_ylabel("künftige Zahlungen je Kalenderjahr (T€)"); ax.legend(); ax.set_title("Fan Chart: erwartete Auszahlungen aus der Reserve 2026–%d" % yrs[-1])
save(fig, "17_fanchart", "Fan Chart künftiger Reservezahlungen", "Fan Chart / Unsicherheitsband", "Band aus Lognormal-Reserve × Dirichlet-Muster (illustrativ)")

# ---------- 4 Schwere & Tail ----------
sev = cl.gross_incurred_eur.values; sev = sev[sev > 0]
shape, loc, scale = stats.lognorm.fit(sev, floc=0)
R["sev_ln"] = {"sigma": shape, "mu": math.log(scale), "mean_emp": float(sev.mean()), "max": float(sev.max())}
u = np.quantile(sev, 0.9); exc = sev[sev > u] - u
gp = stats.genpareto.fit(exc, floc=0); R["gpd"] = {"u": float(u), "xi": gp[0], "beta": gp[2]}
fig, axs = plt.subplots(1, 3, figsize=(13.5, 3.8))
lb = np.logspace(np.log10(sev.min()), np.log10(sev.max()), 40)
axs[0].hist(sev, bins=lb, density=True, color=LIGHT, edgecolor="white"); axs[0].set_xscale("log")
xx = np.logspace(np.log10(sev.min()), np.log10(sev.max()), 300)
axs[0].plot(xx, stats.lognorm.pdf(xx, shape, 0, scale), color=NAVY, lw=2, label="Lognormal-Anpassung"); axs[0].legend(); axs[0].set_title("Histogramm Schadenhöhe"); axs[0].set_xlabel("€ (log)"); axs[0].set_yticks([])
th = stats.lognorm.ppf((np.arange(1, len(sev) + 1) - 0.5) / len(sev), shape, 0, scale)
axs[1].loglog(th, np.sort(sev), ".", color=TEAL); axs[1].plot([th.min(), th.max()], [th.min(), th.max()], "--", color=GREY)
axs[1].set_title("QQ-Plot gegen Lognormal"); axs[1].set_xlabel("theoretisch (€)"); axs[1].set_ylabel("empirisch (€)")
us = np.quantile(sev, np.linspace(0.3, 0.97, 40)); me = [np.mean(sev[sev > v] - v) for v in us]
axs[2].plot(us, me, "o-", color=ORANGE, ms=3); axs[2].set_title("Mean-Excess-Plot"); axs[2].set_xlabel("Schwelle u (€)"); axs[2].set_ylabel("mittlerer Exzess e(u) (€)")
save(fig, "18_schwere_tail", "Schadenhöhe: Histogramm, QQ-Plot, Mean Excess", "Verteilungsdiagnostik")

# ---------- 5 Monte Carlo Gesamtschaden, XL, EP-Kurve ----------
N = 20000
p24 = p[p.policy_year == 2024]
lam = {pr: float(p24[p24["product"] == pr].mu.sum()) for pr in ["motor", "property"]}
sevpar = {}
for pr in ["motor", "property"]:
    s = cl[(cl["product"] == pr) & (cl.cat_event_id.isna())].gross_incurred_eur.values
    sh, _, scl = stats.lognorm.fit(s, floc=0); sevpar[pr] = (sh, scl)
R["mc_lambda"] = lam; R["mc_sev"] = {k: [float(a), float(math.log(b))] for k, (a, b) in sevpar.items()}
att = dict(zip(xl["product"], xl.attachment_eur)); lim = dict(zip(xl["product"], xl.limit_eur))
cat_claims = c[c.cat_event_id.notna()]
cat_n_mean = float(cat_claims.groupby("cat_event_id").size().mean()); cat_rate = len(cat) / 6
R["cat"] = {"events_per_year": cat_rate, "claims_per_event": cat_n_mean, "event_loss": {k: float(v) for k, v in cat_claims.groupby("cat_event_id").gross_incurred_eur.sum().items()}}
def simulate(infl=0.0, catm=1.0, lapse=0.0, n=N, fq=0.0):
    G = np.zeros(n); Nt = np.zeros(n)
    for pr in ["motor", "property"]:
        k = rng.poisson(lam[pr] * (1 - lapse) * (1 + fq), n); sh, scl = sevpar[pr]
        x = rng.lognormal(math.log(scl), sh, k.sum()) * (1 + infl)
        ced = np.clip(x - att[pr], 0, lim[pr]); idx = np.repeat(np.arange(n), k)
        G += np.bincount(idx, x, n); Nt += np.bincount(idx, ced, n)
    # Kumulereignisse (Sach): Anzahl Ereignisse Poisson, Schäden je Ereignis Poisson, Schwere × Multiplikator
    ne = rng.poisson(cat_rate, n); kc = rng.poisson(cat_n_mean * catm, ne.sum()) if ne.sum() else np.array([], int)
    ev_idx = np.repeat(np.repeat(np.arange(n), ne), kc)
    sh, scl = sevpar["property"]; x = rng.lognormal(math.log(scl), sh, kc.sum()) * 2.8 * (1 + infl)
    ced = np.clip(x - att["property"], 0, lim["property"])
    G += np.bincount(ev_idx, x, n); Nt += np.bincount(ev_idx, ced, n)
    return G, G - Nt
G, Nn = simulate()
def risk(v, q=0.995): var = np.quantile(v, q); return float(var), float(v[v >= var].mean())
R["mc"] = {"gross_mean": float(G.mean()), "net_mean": float(Nn.mean()), "gross_var995": risk(G)[0], "gross_tvar995": risk(G)[1], "net_var995": risk(Nn)[0], "net_tvar995": risk(Nn)[1], "prem24": float(p24.gross_premium_eur.sum()), "N": N}
fig, axs = plt.subplots(1, 2, figsize=(12, 4))
axs[0].hist(G / 1000, bins=100, color=LIGHT, edgecolor="white", density=True, label="brutto")
axs[0].hist(Nn / 1000, bins=100, histtype="step", color=TEAL, lw=1.6, density=True, label="netto nach XL")
for v, lab, colr in [(R["mc"]["gross_mean"], "Erwartungswert", NAVY), (R["mc"]["gross_var995"], "VaR 99,5 %", RED), (R["mc"]["gross_tvar995"], "TVaR 99,5 %", ORANGE)]:
    axs[0].axvline(v / 1000, color=colr, ls="--"); axs[0].text(v / 1000, axs[0].get_ylim()[1] * (0.85 if "T" in lab else 0.7), f" {lab}\n {teur(v)} T€", fontsize=8, color=colr)
axs[0].set_xlabel("Jahresgesamtschaden 2026 (T€)"); axs[0].set_yticks([]); axs[0].legend(loc="center right"); axs[0].set_title("Verteilung Jahresgesamtschaden")
rp = np.array([2, 5, 10, 20, 50, 100, 200, 500, 1000])
axs[1].plot(rp, np.quantile(G, 1 - 1 / rp) / 1000, "o-", color=NAVY, lw=2, label="brutto (AEP)")
axs[1].plot(rp, np.quantile(Nn, 1 - 1 / rp) / 1000, "s-", color=TEAL, lw=2, label="netto (AEP)")
axs[1].set_xscale("log"); axs[1].set_xticks(rp); axs[1].set_xticklabels(rp); axs[1].set_xlabel("Wiederkehrperiode in Jahren"); axs[1].set_ylabel("Jahresschaden (T€)")
axs[1].axvline(200, color=RED, ls=":"); axs[1].text(205, axs[1].get_ylim()[0] * 1.02 + 10, "1-in-200", color=RED, fontsize=8); axs[1].legend(); axs[1].set_title("Überschreitungskurve (Exceedance Probability)")
save(fig, "19_gesamtschaden_ep", "Gesamtschadenverteilung und Überschreitungskurve", "Dichte + EP-Kurve", f"Monte Carlo, {N:,} Läufe, Bestand 2024 als Proxy".replace(",", "."))

# XL-Schichtdiagramm
fig, ax = plt.subplots(figsize=(10, 4))
cc = c[c.gross_incurred_eur > 0].sort_values("gross_incurred_eur")
for pr, mk, colr in [("motor", "o", NAVY), ("property", "s", ORANGE)]:
    s = cc[cc["product"] == pr]; ax.scatter(np.arange(len(s)), s.gross_incurred_eur, s=10, marker=mk, color=colr, alpha=0.7, label={"motor": "Kraftfahrt", "property": "Sach"}[pr])
for pr, colr in [("motor", NAVY), ("property", ORANGE)]:
    ax.axhline(att[pr], color=colr, ls="--", lw=1); ax.text(5, att[pr] * 1.08, f"Priorität {pr.replace('motor','Kraftfahrt').replace('property','Sach')}: {eur(att[pr])} €", color=colr, fontsize=8)
ax.set_yscale("log"); ax.set_xlabel("Schäden je Sparte, nach Höhe sortiert"); ax.set_ylabel("Schadenaufwand € (log)")
nce = int((c.ceded_incurred_eur > 0).sum()); R["n_ceded_claims"] = nce
ax.set_title(f"Excess-of-Loss: nur {nce} von {len(c)} Schäden erreichen die Priorität"); ax.legend(loc="center right")
save(fig, "20_xl_schicht", "Schäden relativ zur XL-Priorität", "Sortiertes Streudiagramm mit Schichtgrenzen")

# Marimekko Prämie nach Region × Sparte mit Schadenquote
mm = p.groupby(["region", "product"]).agg(prem=("gross_premium_eur", "sum"), inc=("inc", "sum")).reset_index()
tot_r = mm.groupby("region").prem.sum(); xpos = 0
fig, ax = plt.subplots(figsize=(10, 4.4))
for r in tot_r.sort_values(ascending=False).index:
    wdt = tot_r[r] / tot_r.sum(); ypos = 0
    for prod, colr in [("motor", NAVY), ("property", TEAL)]:
        row = mm[(mm.region == r) & (mm["product"] == prod)].iloc[0]; hgt = row.prem / tot_r[r]
        ax.add_patch(Rectangle((xpos, ypos), wdt, hgt, facecolor=colr, edgecolor="white", lw=2))
        ax.text(xpos + wdt / 2, ypos + hgt / 2, f"SQ {row.inc/row.prem*100:.0f} %", ha="center", va="center", color="white", fontsize=8.5, fontweight="bold")
        ypos += hgt
    ax.text(xpos + wdt / 2, 1.02, f"{r.capitalize()}\n{wdt*100:.0f} %", ha="center", fontsize=8.5); xpos += wdt
ax.set_xlim(0, 1); ax.set_ylim(0, 1.12); ax.grid(False); ax.set_xticks([]); ax.set_ylabel("Anteil Sparte an Regionsprämie")
ax.legend(handles=[Rectangle((0, 0), 1, 1, color=NAVY), Rectangle((0, 0), 1, 1, color=TEAL)], labels=["Kraftfahrt", "Sach"], loc="lower right", bbox_to_anchor=(1, -0.1), ncol=2)
ax.set_title("Marimekko: Prämienvolumen nach Region × Sparte mit Schadenquote (Incurred)", pad=28)
save(fig, "21_marimekko", "Marimekko Prämie und Schadenquote", "Marimekko-/Mosaikdiagramm")

# Schadenquote je Jahr (Incurred + Ultimate CL)
fig, ax = plt.subplots(figsize=(9, 3.8))
ax.bar(AYS, resv.Incurred / resv["Prämie"] * 100, color=LIGHT, edgecolor=GREY, label="gemeldet (Incurred)")
ax.plot(AYS, resv.CL_inc / resv["Prämie"] * 100, "o-", color=NAVY, lw=2, label="Endschaden CL")
ax.plot(AYS, resv.Truth / resv["Prämie"] * 100, "d--", color=ORANGE, lw=1.5, label="Referenz")
ax.axhline(R["elr_apriori"] * 100, color=TEAL, ls=":", label=f"a-priori-SQ BF {R['elr_apriori']*100:.0f} %")
ax.set_ylabel("Schadenquote in % der Bruttoprämie"); ax.legend(ncol=2, fontsize=8); ax.set_title("Schadenquote je Anfalljahr: 2022 durch Sturmkumul erhöht")
save(fig, "22_schadenquote", "Schadenquote je Anfalljahr", "Säulen + Linien")
R["lr_by_year"] = {int(a): round(float(v) * 100, 1) for a, v in (resv.CL_inc / resv["Prämie"]).items()}

# ---------- 6 Markt / ALM ----------
mc["valuation_date"] = pd.to_datetime(mc.valuation_date)
fig, axs = plt.subplots(1, 2, figsize=(12, 3.8))
for i, dt in enumerate(["2019-12-01", "2021-12-01", "2022-12-01", "2023-12-01", "2025-12-01"]):
    s = mc[mc.valuation_date == dt]; axs[0].plot(s.maturity_years, s.risk_free_rate_pct, "o-", color=PAL[i], label=dt[:4])
axs[0].set_xlabel("Laufzeit (Jahre)"); axs[0].set_ylabel("risikofreier Zins %"); axs[0].legend(ncol=5, fontsize=8); axs[0].set_title("Zinsstrukturkurven (synthetisch)")
for i, mt in enumerate([1, 5, 10, 20]):
    s = mc[mc.maturity_years == mt]; axs[1].plot(s.valuation_date, s.risk_free_rate_pct, color=PAL[i], label=f"{mt} J.")
axs[1].set_ylabel("%"); axs[1].legend(ncol=4, fontsize=8); axs[1].set_title("Zeitreihe nach Laufzeit")
save(fig, "23_zinskurve", "Zinsstrukturkurve und Zeitreihe", "Linien / Zeitreihe", "Quelle: synthetisch, nicht EIOPA")
curve = mc[mc.valuation_date == mc.valuation_date.max()].set_index("maturity_years").risk_free_rate_pct
def disc(years, shift_bp=0):
    r = np.interp(years, curve.index, curve.values) / 100 + shift_bp / 10000
    return (1 + r) ** (-years)
tt = np.arange(len(cf)) + 0.5
R["bel_disc"] = float((cf * disc(tt)).sum()); R["duration"] = float((tt * cf * disc(tt)).sum() / (cf * disc(tt)).sum())
fig, ax = plt.subplots(figsize=(9, 3.6))
ax.bar(np.array(yrs) - 0.2, cf / 1000, 0.4, color=LIGHT, edgecolor=GREY, label="nominal")
ax.bar(np.array(yrs) + 0.2, cf * disc(tt) / 1000, 0.4, color=TEAL, label="Barwert")
ax.set_xticks(yrs); ax.set_ylabel("T€"); ax.legend(); ax.set_title(f"Auszahlungsprofil der Reserve: Duration {R['duration']:.2f} Jahre")
save(fig, "24_cashflow_barwert", "Zahlungsstrom nominal vs. Barwert", "Liquiditätsleiter / Säulen")

# ---------- 7 Stress & Kapital ----------
prem26 = R["mc"]["prem24"]
def result(infl=0, rate_bp=0, catm=1, lapse=0, n=8000, fq=0):
    g, nt = simulate(infl, catm, lapse, n, fq)
    # Reserveeffekt: Inflation auf künftige Zahlungen, Diskontierung
    res = (cf * (1 + infl) * disc(tt, rate_bp)).sum()
    return prem26 * (1 - lapse) - nt.mean() - (res - R["bel_disc"]), nt, res
base_r, base_nt, _ = result()
scen = {}
for _, s in sc[sc.scenario_type != "reverse_stress_template"].iterrows():
    r_, nt_, res_ = result(s.claims_inflation_pct / 100, s.rate_shift_bp, s.catastrophe_multiplier, s.lapse_increase_pct / 100)
    scen[s.scenario_id] = {"name": s.scenario_name, "result": float(r_), "net_var995": risk(nt_)[0], "reserve_disc": float(res_)}
R["scen"] = scen
# Wasserfall Basis -> kombiniert (schrittweise)
steps = [("Basis", dict()), ("Schadeninflation +8 %", dict(infl=0.08)), ("Zins +150 bp", dict(infl=0.08, rate_bp=150)), ("Kumul ×2", dict(infl=0.08, rate_bp=150, catm=2)), ("Storno +20 %", dict(infl=0.08, rate_bp=150, catm=2, lapse=0.2))]
vals = []
for lab, kw in steps:
    rng_state = rng.bit_generator.state
    r_, _, _ = result(n=20000, **kw); vals.append(r_)
wf = [vals[0]] + [vals[i] - vals[i - 1] for i in range(1, len(vals))]
R["waterfall"] = {steps[i][0]: float(wf[i]) for i in range(len(wf))}; R["waterfall_end"] = float(vals[-1])
fig, ax = plt.subplots(figsize=(10, 4.2))
cum = 0
for i, (lab, v) in enumerate(zip([s[0] for s in steps], wf)):
    if i == 0: ax.bar(i, v / 1000, color=NAVY); cum = v
    else:
        ax.bar(i, v / 1000, bottom=cum / 1000, color=TEAL if v > 0 else RED); cum += v
    ax.text(i, (cum if i else v) / 1000 + (8 if v >= 0 else -18), f"{'+' if (v>0 and i) else ''}{teur(v)}", ha="center", fontsize=9)
ax.bar(len(steps), cum / 1000, color=ORANGE); ax.text(len(steps), cum / 1000 + 8, teur(cum), ha="center", fontsize=9, fontweight="bold")
ax.set_xticks(range(len(steps) + 1)); ax.set_xticklabels([s[0] for s in steps] + ["Kombinierter\nStress"], fontsize=8.5)
ax.axhline(0, color="black", lw=0.8); ax.set_ylabel("vt. Ergebnis vor Kosten (T€)"); ax.set_ylim(min(0, min(vals)) / 1000 - 40, max(vals) / 1000 + 35)
ax.set_title("Überleitung Basis → kombinierter Stress (erwartetes Nettoergebnis 2026)")
save(fig, "25_wasserfall", "Wasserfall Basis zu kombiniertem Stress", "Wasserfall-/Brückendiagramm", "ohne Kosten, Steuern, Kapitalanlageergebnis")

# Tornado
fac = [("Schadeninflation ±10 %", dict(infl=0.10), dict(infl=-0.10)), ("Zins ±200 bp", dict(rate_bp=200), dict(rate_bp=-200)), ("Kumul ×3 / ×0", dict(catm=3), dict(catm=0)), ("Storno +20 % / −20 %", dict(lapse=0.2), dict(lapse=-0.2)), ("Frequenz +10 % / −10 %", dict(fq=0.10), dict(fq=-0.10))]
torn = []
for lab, up, dn in fac:
    a = result(n=20000, **up)[0] - base_r; bb = result(n=20000, **dn)[0] - base_r; torn.append((lab, a, bb))
torn.sort(key=lambda z: max(abs(z[1]), abs(z[2])))
R["tornado"] = {l: [float(a), float(b_)] for l, a, b_ in torn}
fig, ax = plt.subplots(figsize=(9, 3.8))
for i, (lab, a, bb) in enumerate(torn):
    ax.barh(i, a / 1000, color=RED if a < 0 else TEAL); ax.barh(i, bb / 1000, color=RED if bb < 0 else TEAL, alpha=0.55)
ax.set_yticks(range(len(torn))); ax.set_yticklabels([t[0] for t in torn]); ax.axvline(0, color="black")
ax.set_xlabel("Veränderung erwartetes Ergebnis ggü. Basis (T€)"); ax.set_title("Tornado: Kumulannahme dominiert, gefolgt von Schadeninflation")
save(fig, "26_tornado", "Tornado-Sensitivitätsdiagramm", "Tornado-Diagramm", "kräftig = adverse/erste Richtung, hell = Gegenrichtung")

# Szenariomatrix + Reverse Stress
infls = np.array([0, 2, 4, 6, 8, 10, 12, 15]); cms = np.array([0, 1, 2, 3, 4, 5])
M = np.zeros((len(cms), len(infls)))
for i, cm_ in enumerate(cms):
    for j, inf_ in enumerate(infls): M[i, j] = result(infl=inf_ / 100, catm=cm_, n=4000)[0]
R["matrix"] = {"infl": infls.tolist(), "catm": cms.tolist(), "M": M.round(0).tolist()}
fig, axs = plt.subplots(1, 2, figsize=(13, 4.2))
vmax = np.abs(M).max(); im = axs[0].imshow(M / 1000, cmap=DIV, vmin=-vmax / 1000, vmax=vmax / 1000, aspect="auto", origin="lower")
for i in range(M.shape[0]):
    for j in range(M.shape[1]): axs[0].text(j, i, teur(M[i, j]), ha="center", va="center", fontsize=7.5)
axs[0].set_xticks(range(len(infls))); axs[0].set_xticklabels([f"{v} %" for v in infls]); axs[0].set_yticks(range(len(cms))); axs[0].set_yticklabels([f"×{v}" for v in cms])
axs[0].set_xlabel("Schadeninflation"); axs[0].set_ylabel("Kumul-Multiplikator"); axs[0].grid(False); axs[0].set_title("Szenariomatrix: erwartetes Ergebnis (T€)")
fig.colorbar(im, ax=axs[0], label="T€")
cs = axs[1].contourf(infls, cms, M / 1000, levels=14, cmap=DIV, vmin=-vmax / 1000, vmax=vmax / 1000)
c0 = axs[1].contour(infls, cms, M, levels=[0], colors=RED, linewidths=2.5)
axs[1].clabel(c0, fmt={0: "Ergebnis = 0"}, fontsize=8)
axs[1].set_xlabel("Schadeninflation %"); axs[1].set_ylabel("Kumul-Multiplikator"); axs[1].set_title("Reverse Stress: Kombinationen mit Ergebnis < 0")
save(fig, "27_szenariomatrix_reverse", "Szenariomatrix und Reverse-Stress-Kontur", "Heatmap + Konturplot")

# Standardformel-Logik Prämien- und Reserverisiko (illustrativ)
mot = c[c["product"] == "motor"]; share_liab = float(mot[mot.claim_type == "liability"].gross_incurred_eur.sum() / mot.gross_incurred_eur.sum())
res_by_prod = {}
for pr in ["motor", "property"]:
    ids = c[c["product"] == pr]
    res_by_prod[pr] = float(ids.gross_incurred_eur.sum() - ids.gross_paid_cumulative_eur.sum())
ibnr_tot = R["tot"]["IBNR_CL"]; inc_tot = res_by_prod["motor"] + res_by_prod["property"]
prem_p = p24.groupby("product").gross_premium_eur.sum()
seg = {"Kfz-Haftpflicht": (prem_p["motor"] * share_liab, res_by_prod["motor"] * share_liab, 0.10, 0.09),
       "Sonstige Kfz": (prem_p["motor"] * (1 - share_liab), res_by_prod["motor"] * (1 - share_liab), 0.08, 0.08),
       "Feuer/Sach": (prem_p["property"], res_by_prod["property"], 0.08, 0.10)}
names = list(seg); V, S = [], []
for k in names:
    vp, vr, sp, sr = seg[k]; vr = max(vr, 0) * (1 + ibnr_tot / max(inc_tot, 1))
    v = vp + vr; s = math.sqrt((sp * vp) ** 2 + sp * sr * vp * vr + (sr * vr) ** 2) / v; V.append(v); S.append(s)
V, S = np.array(V), np.array(S)
Corr = np.array([[1, .5, .25], [.5, 1, .25], [.25, .25, 1]])
sig = math.sqrt((S * V) @ Corr @ (S * V)) / V.sum(); scr_pr = 3 * sig * V.sum(); stand = 3 * S * V
R["scr"] = {"segments": {n: {"V": float(v), "sigma": float(s), "SCR_standalone": float(st)} for n, v, s, st in zip(names, V, S, stand)}, "sigma_total": sig, "SCR_prem_res": scr_pr, "div": float(stand.sum() - scr_pr), "share_liab": share_liab}
ne = rng.poisson(cat_rate, 200000); kc = rng.poisson(cat_n_mean, ne.sum()); ev_idx = np.repeat(np.repeat(np.arange(200000), ne), kc)
sh_, scl_ = sevpar["property"]; xc = rng.lognormal(math.log(scl_), sh_, kc.sum()) * 2.8; xc = xc - np.clip(xc - att["property"], 0, lim["property"])
catL = np.bincount(ev_idx, xc, 200000)
R["scr"]["cat_proxy"] = float(np.quantile(catL, 0.995) - catL.mean())
fig, axs = plt.subplots(1, 2, figsize=(12, 4), gridspec_kw={"width_ratios": [1.6, 1]})
xb = list(range(len(names) + 2)); cum = 0
for i, (n_, v) in enumerate(zip(names, stand)):
    axs[0].bar(i, v / 1000, bottom=cum / 1000, color=PAL[i]); axs[0].text(i, (cum + v) / 1000 + 4, teur(v), ha="center", fontsize=8.5); cum += v
axs[0].bar(3, -(stand.sum() - scr_pr) / 1000, bottom=stand.sum() / 1000, color=GREY); axs[0].text(3, stand.sum() / 1000 + 4, f"−{teur(stand.sum()-scr_pr)}", ha="center", fontsize=8.5)
axs[0].bar(4, scr_pr / 1000, color=ORANGE); axs[0].text(4, scr_pr / 1000 + 4, teur(scr_pr), ha="center", fontsize=9, fontweight="bold")
axs[0].set_ylim(0, stand.sum() / 1000 * 1.15); axs[0].set_xticks(xb); axs[0].set_xticklabels(names + ["Diversifikation", "SCR Prämie+\nReserve"], fontsize=8.5); axs[0].set_ylabel("T€")
axs[0].set_title("Prämien- und Reserverisiko nach Standardformel-Logik (illustrativ)")
im = axs[1].imshow(Corr, cmap=SEQ, vmin=0, vmax=1)
for i in range(3):
    for j in range(3): axs[1].text(j, i, f"{Corr[i,j]:.2f}", ha="center", va="center", color="white" if Corr[i, j] > .6 else NAVY)
axs[1].set_xticks(range(3)); axs[1].set_xticklabels(names, rotation=20, fontsize=8); axs[1].set_yticks(range(3)); axs[1].set_yticklabels(names, fontsize=8); axs[1].grid(False); axs[1].set_title("Korrelationsmatrix Segmente")
save(fig, "28_scr_standardformel", "SCR Prämien-/Reserverisiko und Korrelationsmatrix", "Wasserfall + Korrelations-Heatmap", "σ-Faktoren nach DVO (EU) 2015/35 Anh. II (Ursprungsfassung) – vor Nutzung aktuelle Fassung prüfen")

# Solvenzquoten-Pfad (Annahme Eigenmittel)
OF0 = 1_500_000; R["own_funds_assumption"] = OF0
scr_tot = math.sqrt(scr_pr ** 2 + R["scr"]["cat_proxy"] ** 2 + 2 * 0.25 * scr_pr * R["scr"]["cat_proxy"])
R["scr_total_nl"] = scr_tot
fig, ax = plt.subplots(figsize=(9.5, 3.8))
yrs3 = [2025, 2026, 2027, 2028]
for sid, colr in [("BASE", NAVY), ("ST-CLAIMS-INFLATION", ORANGE), ("ST-CAT", TEAL), ("ST-COMBINED", RED)]:
    rr_ = scen[sid]["result"]; of = [OF0] + [OF0 + rr_ * k * 0.5 for k in range(1, 4)]
    ax.plot(yrs3, np.array(of) / scr_tot * 100, "o-", color=colr, lw=2, label=scen[sid]["name"])
ax.axhline(100, color="black", ls="--", lw=1); ax.text(2025.02, 103, "100 % SCR", fontsize=8)
ax.set_xticks(yrs3); ax.set_ylabel("Bedeckungsquote in %"); ax.legend(fontsize=8, ncol=2)
ax.set_title("Solvenzquoten-Pfad je Szenario (Annahme: Eigenmittel 1,5 Mio. €, 50 % Thesaurierung)")
save(fig, "29_solvenzpfad", "Projektion der Bedeckungsquote", "Trajektorie / Zeitreihe", "Eigenmittel und Ausschüttung sind Annahmen, nicht aus den Daten")

# ---------- 8 Geo / Kumul / Governance ----------
pos = {"north": (1, 2), "west": (0, 1), "central": (1, 1), "east": (2, 1), "south": (1, 0)}
reg = p.groupby("region").agg(e=("exposure_years", "sum"), prem=("gross_premium_eur", "sum"), inc=("inc", "sum"))
fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
for ax, col, lab, cmap_ in [(axs[0], "e", "Exposition (PJ)", SEQ), (axs[1], "lr", "Schadenquote %", LinearSegmentedColormap.from_list("o", ["#FFF4E6", ORANGE, RED]))]:
    vals_ = reg.e if col == "e" else reg.inc / reg.prem * 100
    nrm = plt.Normalize(vals_.min() * 0.9, vals_.max());
    for r_, (x_, y_) in pos.items():
        ax.add_patch(Rectangle((x_, y_), 0.95, 0.95, facecolor=cmap_(nrm(vals_[r_])), edgecolor="white", lw=2))
        ax.text(x_ + 0.475, y_ + 0.475, f"{r_.capitalize()}\n{vals_[r_]:,.0f}".replace(",", "."), ha="center", va="center", fontsize=9, color="white" if nrm(vals_[r_]) > 0.6 else NAVY, fontweight="bold")
    ax.set_xlim(-0.1, 3.05); ax.set_ylim(-0.1, 3.05); ax.set_aspect("equal"); ax.axis("off"); ax.set_title(lab)
for _, e_ in cat.iterrows():
    x_, y_ = pos[e_.region]; axs[1].text(x_ + 0.475, y_ + 0.12, "Kumul " + e_.event_date[:4], fontsize=7.5, ha="center", va="center", color="white")
fig.suptitle("Kachelkarte Regionen: Exposition und Schadenquote (mit Jahr des synthetischen Kumulereignisses)", x=0.01, ha="left", fontweight="bold")
save(fig, "30_kachelkarte", "Regionale Kachelkarte Exposition / Schadenquote", "Kachel-/Geokarte")

# Kumulereignisse
ev = cat_claims.groupby("cat_event_id").agg(n=("product", "size"), loss=("gross_incurred_eur", "sum"), ced=("ceded_incurred_eur", "sum"))
fig, ax = plt.subplots(figsize=(9, 3.4))
lbls = [i.replace("CAT-", "").replace("-", " ") for i in ev.index]
ax.barh(lbls, (ev.loss - ev.ced) / 1000, color=NAVY, label="Selbstbehalt"); ax.barh(lbls, ev.ced / 1000, left=(ev.loss - ev.ced) / 1000, color=ORANGE, label="zediert")
for i, (n_, l_) in enumerate(zip(ev.n, ev.loss)): ax.text(l_ / 1000 + 1, i, f"{teur(l_)} T€ · {n_} Schäden", va="center", fontsize=8.5)
ax.set_xlabel("Ereignisschaden brutto (T€)"); ax.legend(loc="lower right"); ax.set_title("Kumulereignisse: Pro-Risiko-XL entlastet Ereignisse kaum")
save(fig, "31_kumulereignisse", "Kumulereignisse brutto/netto", "Gestapelte Balken horizontal")
R["events"] = {k: {"n": int(v.n), "loss": float(v.loss), "ced": float(v.ced)} for k, v in ev.iterrows()}

# Risiko-Heatmap qualitativ
risks = [("Schadeninflation", 4, 4), ("Naturgefahren-Kumul", 2, 5), ("Reserveunterdeckung", 3, 3), ("Zinsänderung", 3, 2), ("RV-Ausfall", 1, 3), ("Datenqualität", 3, 3), ("Modellrisiko", 2, 3), ("Cyber/IT-Ausfall", 2, 4)]
fig, ax = plt.subplots(figsize=(7, 5))
for i in range(5):
    for j in range(5):
        sc_ = (i + 1) * (j + 1); colr = TEAL if sc_ <= 4 else (GOLD if sc_ <= 9 else (ORANGE if sc_ <= 14 else RED))
        ax.add_patch(Rectangle((j + 0.5, i + 0.5), 1, 1, facecolor=colr, alpha=0.35, edgecolor="white"))
jit = {}
for n_, e_, a_ in risks:
    k = (e_, a_); jit[k] = jit.get(k, 0) + 1; off_ = (jit[k] - 1) * 0.22
    ax.plot(e_, a_ + off_ - 0.1, "o", color=NAVY, ms=7); ax.text(e_ + 0.1, a_ + off_ - 0.1, n_, fontsize=8, va="center")
ax.set_xlim(0.5, 5.5); ax.set_ylim(0.5, 5.5); ax.set_xticks(range(1, 6)); ax.set_yticks(range(1, 6)); ax.grid(False)
ax.set_xlabel("Eintrittswahrscheinlichkeit (1–5)"); ax.set_ylabel("Auswirkung (1–5)"); ax.set_title("Risiko-Heatmap (qualitative Einschätzung)")
save(fig, "32_risiko_heatmap", "Qualitative Risiko-Heatmap", "Risikomatrix", "Scores sind Expertenschätzung der KI, keine Kapitalmessung")

# Datenqualitäts-Prüfmatrix (aus Defektkatalog)
dc = pd.read_csv(f"{DS}/defect_fixtures/defect_catalog.csv")
R["defects"] = dc.to_dict(orient="records")
rules = dc.validator_code.tolist(); files = sorted(set(dc.file_name))
fig, ax = plt.subplots(figsize=(9, 3.4))
Mq = np.zeros((2, len(rules)))
Mq[1, :] = 1
ax.imshow(Mq, cmap=LinearSegmentedColormap.from_list("q", ["#CFE8E5", "#F3C6CB"]), aspect="auto")
for j, r_ in enumerate(rules):
    ax.text(j, 0, "✓ bestanden", ha="center", va="center", fontsize=8, color=TEAL); ax.text(j, 1, "✗ erkannt", ha="center", va="center", fontsize=8, color=RED)
ax.set_xticks(range(len(rules))); ax.set_xticklabels([f"{r_}\n{f_.replace('.csv','')}" for r_, f_ in zip(rules, dc.file_name)], fontsize=7.5)
ax.set_yticks([0, 1]); ax.set_yticklabels(["Profil clean", "Profil defect_fixtures"]); ax.grid(False)
ax.set_title("Datenqualitäts-Prüfmatrix: alle 6 eingebauten Fehlerklassen werden erkannt")
save(fig, "33_dq_matrix", "Datenqualitäts-Prüfmatrix", "Ampel-/Prüfmatrix")

# Sankey-artige Flussdarstellung Prämie -> Schaden -> brutto/netto
fig, ax = plt.subplots(figsize=(11, 4.4)); ax.axis("off"); ax.grid(False)
P_ = R["premium"]; Ginc = R["tot"]["CL_inc"]; ced_ = float(c.ceded_incurred_eur.sum()); paid_ = R["paid_cut"]; resv_ = Ginc - paid_
sc_ = 1 / P_
def band(x0, y0, h0, x1, y1, h1, colr, a=0.45):
    t_ = np.linspace(0, 1, 50); s_ = 3 * t_ ** 2 - 2 * t_ ** 3
    top = y0 + h0 + (y1 + h1 - y0 - h0) * s_; bot = y0 + (y1 - y0) * s_; xs_ = x0 + (x1 - x0) * t_
    ax.fill_between(xs_, bot, top, color=colr, alpha=a, lw=0)
def node(x, y, h, lab, colr): ax.add_patch(Rectangle((x, y), 0.03, h, color=colr)); ax.text(x + 0.04, y + h / 2, lab, va="center", fontsize=8.5)
node(0.02, 0, 1, f"Bruttoprämie\n{teur(P_)} T€", NAVY)
g_h = Ginc * sc_; m_h = 1 - g_h
band(0.05, 1 - g_h, g_h, 0.38, 0.45 - 0.0, g_h, ORANGE); band(0.05, 0, m_h, 0.38, 0.0 - 0.05, m_h, TEAL)
node(0.38, 0.45, g_h, f"Endschaden CL\n{teur(Ginc)} T€", ORANGE); node(0.38, -0.05, m_h, f"Marge vor Kosten\n{teur(P_-Ginc)} T€", TEAL)
ph = paid_ * sc_; rh = resv_ * sc_
band(0.41, 0.45 + rh, ph, 0.75, 0.62, ph, NAVY); band(0.41, 0.45, rh, 0.75, 0.35, rh, GOLD)
node(0.75, 0.62, ph, f"bereits gezahlt\n{teur(paid_)} T€", NAVY); node(0.75, 0.35, rh, f"Reserve (Einzel + IBNR)\n{teur(resv_)} T€", GOLD)
ax.text(0.75, 0.25, f"davon an Rückversicherer zediert: {teur(ced_)} T€", fontsize=8, color=RED)
ax.set_xlim(0, 1); ax.set_ylim(-0.1, 1.05); ax.set_title("Flussdiagramm (Sankey): Was wird aus 100 € Prämie? (Policenjahre 2019–2024)")
save(fig, "34_sankey", "Sankey-Fluss Prämie → Schaden → Zahlung/Reserve", "Sankey-Diagramm")

# Credibility (Bühlmann-Straub) nach Region
reg_y = p.groupby(["region", "policy_year"]).agg(n=("n", "sum"), e=("exposure_years", "sum")).reset_index(); reg_y["f"] = reg_y.n / reg_y.e
wi = reg_y.groupby("region").e.sum(); fi = reg_y.groupby("region").apply(lambda g: (g.n.sum() / g.e.sum()), include_groups=False)
fbar = p.n.sum() / p.exposure_years.sum()
s2 = (reg_y.e * (reg_y.f - reg_y.region.map(fi)) ** 2).sum() / (len(reg_y) - reg_y.region.nunique())
m_ = wi.sum(); a_ = max(((wi * (fi - fbar) ** 2).sum() - (len(wi) - 1) * s2) / (m_ - (wi ** 2).sum() / m_), 1e-9)
Z = wi / (wi + s2 / a_); cred = Z * fi + (1 - Z) * fbar
R["cred"] = {r_: {"Z": float(Z[r_]), "raw": float(fi[r_]), "cred": float(cred[r_])} for r_ in wi.index}
fig, ax = plt.subplots(figsize=(9, 3.6))
xi = np.arange(len(wi))
ax.bar(xi - 0.2, fi * 100, 0.4, color=LIGHT, edgecolor=GREY, label="beobachtet")
ax.bar(xi + 0.2, cred * 100, 0.4, color=TEAL, label="Credibility-gewichtet")
ax.axhline(fbar * 100, color=NAVY, ls="--", label="Kollektiv")
for i_, z_ in enumerate(Z): ax.text(i_ + 0.2, cred.iloc[i_] * 100 + 0.1, f"Z={z_:.2f}", ha="center", fontsize=8)
ax.set_xticks(xi); ax.set_xticklabels(wi.index.str.capitalize()); ax.set_ylabel("Schäden je 100 PJ"); ax.set_ylim(0, 10.5); ax.legend(ncol=3, fontsize=8, loc="upper left")
ax.set_title("Bühlmann-Straub-Credibility: Regionsfrequenzen zum Kollektiv gezogen")
save(fig, "35_credibility", "Credibility-Gewichtung nach Region", "Säulen mit Credibility-Faktoren")

# Kleine Vielfache: Frequenz je Jahr und Sparte (Trend)
fig, axs = plt.subplots(1, 2, figsize=(11, 3.4), sharey=False)
for ax, pr, colr in [(axs[0], "motor", NAVY), (axs[1], "property", ORANGE)]:
    s = py[py["product"] == pr]; f_ = s.n / s.expo; lo_ = stats.chi2.ppf(0.025, 2 * s.n) / 2 / s.expo; hi_ = stats.chi2.ppf(0.975, 2 * (s.n + 1)) / 2 / s.expo
    ax.fill_between(s.policy_year, lo_ * 100, hi_ * 100, color=colr, alpha=0.15); ax.plot(s.policy_year, f_ * 100, "o-", color=colr, lw=2)
    ax.set_title({"motor": "Kraftfahrt", "property": "Sach"}[pr]); ax.set_ylabel("Schäden je 100 PJ")
fig.suptitle("Frequenztrend je Sparte mit 95 %-Band: kein signifikanter Trend außer Kumuljahren", x=0.01, ha="left", fontweight="bold")
save(fig, "36_frequenztrend", "Frequenztrend je Sparte", "Small Multiples / Zeitreihe mit Band")

json.dump(R, open(os.path.join(ERG, "results.json"), "w"), indent=1, default=float, ensure_ascii=False)
json.dump(CHARTS, open(os.path.join(ERG, "charts.json"), "w"), indent=1, ensure_ascii=False)
print("charts", len(CHARTS))
