"""Nachbau des stochastischen Einzelschadenreservierungsverfahrens (sicr, D. Skowasch) in Python
und rollierender Backtest gegen gängige Marktverfahren auf den Beispieldaten des sicr-Pakets.
Nur Schadenzahlungen (Cl_*); Renten (An_*) bleiben in allen Verfahren außen vor.
"""
import warnings, json, sys, math
warnings.filterwarnings("ignore")
import os
import numpy as np, pandas as pd, rdata

HERE = os.path.dirname(os.path.abspath(__file__))
TEIL = os.path.dirname(HERE)
# Pfad zu den Beispieldaten des R-Pakets sicr (Ordner data/ aus github.com/dskowasch/sicr)
DATA = os.environ.get("SICR_DATA", os.path.join(TEIL, "daten", "sicr", "data"))
ERG = os.path.join(TEIL, "ergebnisse"); os.makedirs(ERG, exist_ok=True)
D = lambda n: rdata.read_rda(f"{DATA}/{n}.rda")[n]
cd = D("claims_data_xmpl")[["Claim_id", "Origin_year", "Calendar_year", "Cl_payment_cal", "Cl_reserve"]].copy()
cd["Origin_year"] = cd.Origin_year.astype(int); cd["Calendar_year"] = cd.Calendar_year.astype(int)
hi = D("historic_indices_xmpl"); re_ = D("reinsurance_xmpl")
lvl = (1 + hi.set_index("Calendar_year").Index_gross).cumprod(); lvl.index = lvl.index.astype(int)

# Parameter wie im Paketbeispiel
TH, IDX_YEAR = 400_000, 2022
RC = np.array([1, 200_001, 400_001, 700_001, 1_400_001])
SOT, EOT = 17, 50
FIRST = 1989
NCLS = len(RC) + 1

def tail_factor(d):
    return np.minimum(1, 1 - np.minimum(1, (d - SOT) / (EOT - SOT + 1)))

def cls(x):
    return np.searchsorted(RC, x, side="right")  # = findInterval

def prepare(K):
    """Daten bis Kalenderjahr K, indexiert auf IDX_YEAR, mit Großschadenkennzeichen."""
    x = cd[cd.Calendar_year <= K]
    meta = x.groupby("Claim_id").Origin_year.first()
    st = np.maximum(meta.values, FIRST - 1)
    full = pd.DataFrame({"Claim_id": np.repeat(meta.index.values, K - st + 1), "Origin_year": np.repeat(meta.values, K - st + 1),
                         "Calendar_year": np.concatenate([np.arange(a, K + 1) for a in st])})
    x = full.merge(x, on=["Claim_id", "Origin_year", "Calendar_year"], how="left").fillna({"Cl_payment_cal": 0.0, "Cl_reserve": 0.0})
    x = x.sort_values(["Claim_id", "Calendar_year"]).copy()
    f = lvl[IDX_YEAR] / x.Calendar_year.map(lvl)
    x["ipay"] = x.Cl_payment_cal * f; x["ires"] = x.Cl_reserve * f
    x["icum"] = x.groupby("Claim_id").ipay.cumsum(); x["iinc"] = x.icum + x.ires
    big = x[x.iinc > TH].groupby("Claim_id").Calendar_year.min().rename("ls")
    x = x.join(big, on="Claim_id"); x = x[x.ls.notna()].copy()
    old = x.Origin_year < FIRST
    x.loc[old, "ls"] = x.loc[old, "Origin_year"] + 3 - 1
    x["dev"] = (x.Calendar_year - x.ls + 1).astype(int)
    x["ires_entry"] = x.groupby("Claim_id").ires.shift(1).fillna(0)
    x.loc[x.Calendar_year == x.Origin_year, "ires_entry"] = 0
    x["c_in"] = cls(x.ires_entry.values); x["c_out"] = cls(x.ires.values)
    return x

def build_pools(x, K, years=10):
    p = x[(x.Calendar_year > K - years) & (x.Calendar_year <= K) & (x.dev >= 2)]
    pay, cout, start, length = [], [], np.zeros((SOT + 1, NCLS), int), np.zeros((SOT + 1, NCLS), int)
    pos = 0
    for c in range(NCLS):
        for d in range(2, SOT + 1):
            if d < SOT: s = p[(p.dev == d) & (p.c_in == c)]; v = s.ipay.values
            else:
                s = p[(p.dev >= SOT) & (p.dev <= EOT) & (p.c_in == c)]; v = s.ipay.values / tail_factor(s.dev.values)
            if len(s) == 0: v = np.array([0.0]); co = np.array([c])
            else: co = s.c_out.values
            pay.append(v); cout.append(co); start[d, c] = pos; length[d, c] = len(v); pos += len(v)
    return np.concatenate(pay), np.concatenate(cout), start, length

def simulate(x, K, pools, n=1000, horizon=EOT, infl=0.03, seed=1):
    """Gibt (n × claims × horizon) nominale künftige Zahlungen zurück."""
    pay, cout, start, length = pools
    last = x[x.Calendar_year == x.groupby("Claim_id").Calendar_year.transform("max")].set_index("Claim_id")
    last = last[last.ls <= K]
    d0 = (K - last.ls + 1).astype(int).values; c0 = last.c_out.values
    m = len(last); rng = np.random.default_rng(seed)
    out = np.zeros((n, m, horizon))
    to_nom = lvl[K] / lvl[IDX_YEAR] * (1 + infl) ** np.arange(1, horizon + 1)
    for s in range(n):
        c = c0.copy()
        for y in range(horizon):
            d = d0 + y + 1; act = d <= EOT
            dd = np.minimum(d, SOT)
            st = start[dd, c]; ln = length[dd, c]
            r = st + (rng.random(m) * ln).astype(int)
            out[s, :, y] = np.where(act, pay[r] * tail_factor(d), 0) * to_nom[y]
            c = np.where(act, cout[r], c)
    return out, last

def cohort_triangle(x, K, col):
    """Kohorte = Jahr des Großwerdens, Abwicklung = Jahr seit Großwerden (nominal)."""
    g = x[x.ls <= K].copy(); g["val"] = g[col]
    T = g.pivot_table(index="ls", columns="dev", values="val", aggfunc="sum").fillna(0)
    T = T.reindex(columns=range(1, T.columns.max() + 1), fill_value=0)
    return T

def cl_paid_future(x, K, h, return_factors=False, tri=None):
    inc = tri if tri is not None else cohort_triangle(x, K, "Cl_payment_cal")
    coh = inc.index.values.astype(int); J = inc.shape[1]
    cum = inc.cumsum(axis=1).values
    obs = np.array([[K - ch + 1 >= j for j in range(1, J + 1)] for ch in coh])
    f = np.ones(J + 60)
    for j in range(J - 1):
        m = obs[:, j + 1] & obs[:, j]
        den = cum[m, j].sum()
        f[j] = cum[m, j + 1].sum() / den if m.sum() and den > 0 else 1.0
    fut = np.zeros(h); allfut = 0.0
    for i, ch in enumerate(coh):
        k = K - ch + 1; c = cum[i, k - 1] if k <= J else cum[i, -1]
        for t in range(1, 80):
            j = k + t
            if j - 2 >= len(f): break
            nc = c * f[j - 2]; incv = nc - c; c = nc
            allfut += incv
            if t <= h: fut[t - 1] += incv
    return (fut, allfut, f) if return_factors else (fut, allfut)

def future_by_pattern(x, K, h, total_res):
    """Verteilt eine Gesamtreserve zeitlich nach dem Paid-CL-Muster."""
    fut, allfut = cl_paid_future(x, K, h)
    return total_res * fut / allfut if allfut else np.zeros(h)

def cl_incurred_reserve(x, K):
    g = x[x.ls <= K].copy(); g["inc"] = g.groupby("Claim_id").Cl_payment_cal.cumsum() + g.Cl_reserve
    T = g.pivot_table(index="ls", columns="dev", values="inc", aggfunc="sum")
    T = T.reindex(columns=range(1, int(T.columns.max()) + 1))
    # fehlende Zellen (Schaden ohne Zeile) mit letztem Stand fortschreiben
    T = T.fillna(0)
    coh = T.index.values.astype(int); J = T.shape[1]; V = T.values
    obs = np.array([[K - ch + 1 >= j for j in range(1, J + 1)] for ch in coh])
    f = np.ones(J)
    for j in range(J - 1):
        m = obs[:, j + 1] & obs[:, j]; den = V[m, j].sum()
        f[j] = V[m, j + 1].sum() / den if m.sum() and den > 0 else 1.0
    ult = 0.0
    for i, ch in enumerate(coh):
        k = min(K - ch + 1, J); ult += V[i, k - 1] * np.prod(f[k - 1:])
    paid = x[x.ls <= K].Cl_payment_cal.sum()
    return ult - paid

def odp_bootstrap(x, K, h, B=1000, seed=3):
    inc = cohort_triangle(x, K, "Cl_payment_cal"); coh = inc.index.values.astype(int)
    J = inc.shape[1]; I = len(coh)
    obs = np.array([[K - ch + 1 >= j for j in range(1, J + 1)] for ch in coh])
    _, _, f = cl_paid_future(x, K, h, True)
    cum = inc.cumsum(axis=1).values
    # angepasste Werte rückwärts von der Diagonale
    fit = np.zeros_like(cum)
    for i, ch in enumerate(coh):
        k = min(K - ch + 1, J); fit[i, k - 1] = cum[i, k - 1]
        for j in range(k - 2, -1, -1): fit[i, j] = fit[i, j + 1] / f[j] if f[j] else 0
    fitinc = np.diff(np.c_[np.zeros(I), fit], axis=1)
    incv = inc.values
    msk = obs & (np.abs(fitinc) > 1)
    res = (incv[msk] - fitinc[msk]) / np.sqrt(np.abs(fitinc[msk]))
    nobs, npar = msk.sum(), I + J - 1
    phi = (res ** 2).sum() / max(nobs - npar, 1); res = res * math.sqrt(nobs / max(nobs - npar, 1))
    rng = np.random.default_rng(seed); out = np.zeros((B, h))
    for b in range(B):
        ps = incv.copy()
        ps[msk] = fitinc[msk] + rng.choice(res, msk.sum()) * np.sqrt(np.abs(fitinc[msk]))
        ps[~obs] = 0
        fut, _ = cl_paid_future(x, K, h, tri=pd.DataFrame(ps, index=inc.index, columns=inc.columns))
        # Prozessvarianz (Gamma mit Skalenparameter phi), vorzeichengerecht
        mu = np.abs(fut); sh = np.where(mu > 0, mu / phi, 0)
        proc = np.where(mu > 0, rng.gamma(np.maximum(sh, 1e-9), phi), 0) * np.sign(fut)
        out[b] = proc
    return out

def actual_future(K, h, ids):
    a = cd[(cd.Claim_id.isin(ids)) & (cd.Calendar_year > K) & (cd.Calendar_year <= K + h)]
    return a.groupby("Calendar_year").Cl_payment_cal.sum().reindex(range(K + 1, K + h + 1), fill_value=0).values

def xl_ceded(cum_nom, origin):
    r = re_.set_index(re_.Origin_year.astype(int)).reindex(origin)
    P, L, Q = r.Priority.fillna(0).values, r.Limit.fillna(0).values, r.Quota_share.fillna(0).values
    base = np.maximum(cum_nom, 0) * (1 - Q)[..., None] if cum_nom.ndim == 2 else np.maximum(cum_nom, 0) * (1 - Q)
    if cum_nom.ndim == 2:
        return np.clip(base - P[:, None], 0, L[:, None])
    return np.clip(base - P, 0, L)

# ------------------------------------------------------------------ Backtest
H = 5; N = 400
rows, store = [], {}
for K in range(2008, 2019):
    x = prepare(K); pools = build_pools(x, K)
    infl = float(hi.set_index("Calendar_year").Index_gross.loc[K - 9:K].mean())
    sims, last = simulate(x, K, pools, n=N, horizon=H, infl=infl, seed=K)
    ids = last.index.values
    act = actual_future(K, H, ids)
    s_tot = sims.sum(axis=(1, 2))
    clp, clp_all = cl_paid_future(x, K, H)
    res_inc = cl_incurred_reserve(x, K); cli = future_by_pattern(x, K, H, res_inc)
    case_res = last.Cl_reserve.sum(); cr = future_by_pattern(x, K, H, case_res)
    boot = odp_bootstrap(x, K, H, B=400, seed=K)
    b_tot = boot.sum(axis=1)
    A = act.sum()
    # XL im Fenster: kumulierte Nominalzahlungen bis K plus Fenster
    paid_K = x[x.ls <= K].groupby("Claim_id").Cl_payment_cal.sum().reindex(ids).values
    origin = last.Origin_year.values.astype(int)
    a_cl = cd[(cd.Claim_id.isin(ids)) & (cd.Calendar_year > K) & (cd.Calendar_year <= K + H)].groupby("Claim_id").Cl_payment_cal.sum().reindex(ids, fill_value=0).values
    xl_act = (xl_ceded(paid_K + a_cl, origin) - xl_ceded(paid_K, origin)).sum()
    sim_cum = paid_K[None, :] + sims.sum(axis=2)
    xl_sim = (np.clip(np.maximum(sim_cum, 0) * (1 - re_.set_index(re_.Origin_year.astype(int)).reindex(origin).Quota_share.fillna(0).values)[None, :] - re_.set_index(re_.Origin_year.astype(int)).reindex(origin).Priority.fillna(0).values[None, :], 0, re_.set_index(re_.Origin_year.astype(int)).reindex(origin).Limit.fillna(0).values[None, :]) - xl_ceded(paid_K, origin)[None, :]).sum(axis=1)
    # Brutto-Netto-Quote (historisch): Anteil zedierter XL-Zahlungen an Bruttozahlungen der letzten 10 Jahre
    hist = cd[cd.Claim_id.isin(ids) & (cd.Calendar_year <= K)].sort_values(["Claim_id", "Calendar_year"]).copy()
    hist["cum"] = hist.groupby("Claim_id").Cl_payment_cal.cumsum()
    org = hist.Claim_id.map(last.Origin_year).astype(int).values
    hist["xl"] = xl_ceded(hist.cum.values, org); hist["xl_inc"] = hist.groupby("Claim_id").xl.diff().fillna(hist.xl)
    hw = hist[hist.Calendar_year > K - 10]
    gn_ratio = hw.xl_inc.sum() / hw.Cl_payment_cal.sum()
    rows.append(dict(K=K, n_claims=len(ids), actual=A, sicr=s_tot.mean(), sicr_q05=np.quantile(s_tot, .05), sicr_q95=np.quantile(s_tot, .95),
                     sicr_pit=float((s_tot < A).mean()), cl_paid=clp.sum(), cl_inc=cli.sum(), case=cr.sum(),
                     boot_mean=b_tot.mean(), boot_q05=np.quantile(b_tot, .05), boot_q95=np.quantile(b_tot, .95), boot_pit=float((b_tot < A).mean()),
                     xl_actual=xl_act, xl_sicr=xl_sim.mean(), xl_sicr_q05=np.quantile(xl_sim, .05), xl_sicr_q95=np.quantile(xl_sim, .95),
                     xl_gn=gn_ratio * clp.sum(), case_res_total=case_res, cl_inc_res_total=res_inc, cl_paid_res_total=clp_all,
                     infl=infl))
    store[K] = dict(act=act.tolist(), sicr=sims.sum(axis=1).mean(0).tolist(), sicr_q=[np.quantile(sims.sum(axis=1), q, axis=0).tolist() for q in (.05, .5, .95)],
                    clp=clp.tolist(), cli=cli.tolist(), case=cr.tolist(), s_tot=np.quantile(s_tot, np.linspace(0, 1, 41)).tolist(), b_tot=np.quantile(b_tot, np.linspace(0, 1, 41)).tolist())
    print(K, len(ids), round(A / 1e6, 2), round(s_tot.mean() / 1e6, 2), round(clp.sum() / 1e6, 2), round(cli.sum() / 1e6, 2), round(cr.sum() / 1e6, 2), round(b_tot.mean() / 1e6, 2), "XL", round(xl_act / 1e6, 2), round(xl_sim.mean() / 1e6, 2), round(gn_ratio * clp.sum() / 1e6, 2), flush=True)

bt = pd.DataFrame(rows)
bt.to_csv(os.path.join(ERG, "backtest.csv"), index=False)
json.dump(store, open(os.path.join(ERG, "backtest_detail.json"), "w"))

# ------------------------------------------------------------------ Volle Bewertung 2023
K = 2023; x = prepare(K); pools = build_pools(x, K)
sims, last = simulate(x, K, pools, n=1000, horizon=EOT, infl=0.03, seed=7)
tot = sims.sum(axis=(1, 2))
clp, clp_all = cl_paid_future(x, K, EOT)
full = dict(n_claims=len(last), sicr_mean=float(tot.mean()), sicr_sd=float(tot.std()), sicr_q=[float(np.quantile(tot, q)) for q in (.05, .5, .75, .95, .995)],
            cl_paid_all=float(clp_all), cl_inc_res=float(cl_incurred_reserve(x, K)), case_res=float(last.Cl_reserve.sum()),
            cf_sicr=sims.sum(axis=1).mean(0)[:30].tolist(), cf_clp=clp[:30].tolist(),
            pool_rows=int(len(pools[0])), pool_len=pools[3].tolist(), dist=np.quantile(tot, np.linspace(0.001, 0.999, 200)).tolist())
# Konvergenz: Standardfehler des Mittelwerts in Abhängigkeit von n
full["se_mean_pct"] = float(tot.std() / math.sqrt(len(tot)) / tot.mean() * 100)
# Sensitivität Schwelle / Klassengrenzen / Poolfenster
sens = {}
for lab, th, rc, yrs in [("Basis", 400_000, RC, 10), ("Schwelle 300 T€", 300_000, RC, 10), ("Schwelle 600 T€", 600_000, RC, 10),
                         ("Gröbere Klassen", 400_000, np.array([1, 400_001, 1_400_001]), 10), ("Feinere Klassen", 400_000, np.array([1, 100_001, 200_001, 400_001, 700_001, 1_000_001, 1_400_001, 2_500_001]), 10),
                         ("Poolfenster 5 J.", 400_000, RC, 5), ("Poolfenster 20 J.", 400_000, RC, 20)]:
    TH, RC_b = th, RC; RC = rc; NCLS = len(RC) + 1
    xs = prepare(K); ps = build_pools(xs, K, years=yrs)
    ss, ll = simulate(xs, K, ps, n=300, horizon=EOT, infl=0.03, seed=11)
    sens[lab] = dict(mean=float(ss.sum(axis=(1, 2)).mean()), n=int(len(ll)), case=float(ll.Cl_reserve.sum()))
    TH, RC = 400_000, RC_b; NCLS = len(RC) + 1
full["sens"] = sens
json.dump(full, open(os.path.join(ERG, "full2023.json"), "w"))
print(json.dumps({k: v for k, v in full.items() if k not in ("cf_sicr", "cf_clp", "dist", "pool_len")}, indent=1))
