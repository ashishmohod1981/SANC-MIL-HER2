# ===================== CELL E2. Site adaptation on ADEL with the site's own labels (about 40–70 min on a T4) =====================
# Question: when a negative-constrained model is moved to a new site, can it be adapted with labels the site already has
# (IHC scores, no pixel annotation), and does the localisation advantage survive the adaptation?
#
# PROTOCOL (fixed before running; do not change after seeing results)
#   * Data: all 418 ADEL slides at the primary scale (0.25 µm/px), frozen stain matrix - the same tiles as cell C.
#   * Outer evaluation: 5-fold cross-validation stratified by IHC score, repeated 3 times (seeds 0, 1, 2). Every slide is
#     predicted only by models that never saw it. Inside each training fold, 20% (stratified) is held out as a calibration
#     set for the site threshold and temperature; the remaining 80% is used for fine-tuning.
#   * Fine-tuning: each of the 5 GEA fold models is fine-tuned separately and the 5 are averaged, as in the main analysis.
#     All head parameters, AdamW lr 5e-5, weight decay 1e-3, cosine schedule, 15 epochs, 16 bags per step, <= 256 tiles per bag,
#     10% tile drop, gradient clip 1.0, sampling balanced over IHC 0/1+/2+/3+. Losses exactly as in training:
#       status BCE on 0/1+ (negative) and 3+ (positive) slides (2+ excluded: no ISH), ordinal CORAL loss on all slides (w 0.5),
#       negative-bag loss on every tile of IHC 0/1+ slides (w 0.5; constrained variant only), top-k loss on 3+ slides (w 0.5).
#   * Both variants are adapted identically: SANC-MIL and SANC-MIL without the negative constraint (which keeps neg = False).
#   * Primary end points (Holm-corrected together):
#       P1  false-positive tile fraction in IHC 0 slides, adapted SANC-MIL vs adapted model without the constraint (paired);
#       P2  status AUROC (0/1+ vs 3+) of the SANC-MIL status logit, adapted vs not adapted (paired bootstrap).
#   * Secondary: ordinal-head AUROC; FP in IHC 1+; positive coverage in IHC 3+; QWK; sensitivity / specificity at the GEA
#     threshold and at the site threshold fitted on the calibration set; per-repeat variability.
# Outputs: tables/external_adel_site_adaptation_{per_slide,summary,per_repeat}.csv, a figure, and a zip to send back.
# Run in the same session after cells C and D (needs the section-21 functions and the saved GEA fold models).
# Resumable: tile bags and every (repeat, fold) result are cached.
import copy, math, pickle, zipfile, json, time
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from pathlib import Path
from scipy import stats
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, cohen_kappa_score

NAME, NATIVE = "adel_mpp025", 0.25
REPEATS, OUTER, CAL_FRAC = (0, 1, 2), 5, 0.20
FT = dict(epochs=15, lr=5e-5, wd=1e-3, bs=16, max_tiles=256, drop=0.10)
NOCON = "SANC − negative constraint"
VARS = {"SANC-MIL": PROPOSED, "without constraint": NOCON}
DST = OUT / "ext_adel_regions"
BAGDIR = CACHE / "ext_bags" / NAME; BAGDIR.mkdir(parents=True, exist_ok=True)
RESDIR = CACHE / "ext_adapt" / NAME; RESDIR.mkdir(parents=True, exist_ok=True)
t_start = time.time()

# ---------------- 1. manifest (identical item ids to cell C) ----------------
man = build_manifest(str(DST), NAME, str(DST / f"manifest_native{NATIVE:.2f}.csv"))
man = man[man.ihc_score.notna()].reset_index(drop=True)
man["item_id"] = [re.sub(r"[^A-Za-z0-9]+", "_", Path(p).stem)[:80] + "_" + hashlib.md5(str(p).encode()).hexdigest()[:6] for p in man.path]
man["ihc_score"] = man.ihc_score.astype(int)
print(f"{len(man)} slides | {man.ihc_score.value_counts().sort_index().to_dict()}")

# ---------------- 2. tile bags (embeddings + stain features), same tiles and seeds as cell C ----------------
use_embedding("emb")
@torch.no_grad()
def extract_bag(row, enc):
    rng = np.random.default_rng(_img_seed(row.item_id)); embs, hands = [], []
    for work, I0 in iter_regions(row, rng):
        out = region_tile_features(ENGINE, work, I0)
        if out is None: continue
        mp, hand, coords = out; del mp
        if not len(coords): continue
        xg = torch.from_numpy(np.ascontiguousarray(work)).to(DEVICE).permute(2, 0, 1); T_ = CFG.TILE
        for b in range(0, len(coords), CFG.ENC_BATCH):
            cb = coords[b:b + CFG.ENC_BATCH]
            batch = torch.stack([xg[:, y:y + T_, x:x + T_] for y, x in cb]).float() / 255
            with autocast(): embs.append(gpu_safe(enc, batch).float().cpu().numpy().astype(np.float16))
        hands.append(hand)
    if not hands: return None
    hand, emb = np.concatenate(hands), np.concatenate(embs)
    if len(hand) > CFG.MAX_TILES_PER_IMAGE:
        k = np.sort(rng.choice(len(hand), CFG.MAX_TILES_PER_IMAGE, replace=False)); hand, emb = hand[k], emb[k]
    return dict(emb=emb, hand=hand)

todo = [r for _, r in man.iterrows() if not (BAGDIR / f"{r.item_id}.npz").exists()]
if todo:
    enc = TileEncoder(CFG.ENCODER).to(DEVICE).eval()
    for r in tqdm(todo, desc="caching ADEL tile bags"):
        try: bag = extract_bag(r, enc)
        except Exception as e: print(" ", r.item_id, type(e).__name__, str(e)[:100]); bag = None
        if bag is not None: np.savez(BAGDIR / f"{r.item_id}.npz", **bag)
    del enc; torch.cuda.empty_cache()
man = man[[(BAGDIR / f"{i}.npz").exists() for i in man.item_id]].reset_index(drop=True)
BAGS_EXT = []
for i in man.item_id:
    z = np.load(BAGDIR / f"{i}.npz")
    h = ((transform_hand(z["hand"]) - HAND_MU) / HAND_SD).astype(np.float32)
    BAGS_EXT.append((torch.from_numpy(np.array(z["emb"])).to(DEVICE), torch.from_numpy(h).to(DEVICE)))
SC = man.ihc_score.values; N = len(man)
print(f"bags ready: {N} slides, {sum(len(b[0]) for b in BAGS_EXT)} tiles")

def make_batch(idx, rng=None, max_tiles=None, drop=0.0):
    sel = []
    for b in idx:
        n = len(BAGS_EXT[b][0]); t = np.arange(n)
        if max_tiles and n > max_tiles: t = np.sort(rng.choice(t, max_tiles, replace=False))
        if drop > 0 and len(t) > 4:
            k = rng.random(len(t)) >= drop; t = t[k] if k.any() else t
        sel.append(t)
    L = max(len(t) for t in sel); E = torch.zeros(len(idx), L, BAGS_EXT[0][0].shape[1], device=DEVICE)
    Hd = torch.zeros(len(idx), L, BAGS_EXT[0][1].shape[1], device=DEVICE); M = torch.zeros(len(idx), L, dtype=torch.bool, device=DEVICE)
    for r, (b, t) in enumerate(zip(idx, sel)):
        tt = torch.from_numpy(t).to(DEVICE)
        E[r, :len(t)] = BAGS_EXT[b][0][tt].float(); Hd[r, :len(t)] = BAGS_EXT[b][1][tt]; M[r, :len(t)] = True
    return E, Hd, M

@torch.no_grad()
def predict(models, idx, T):
    out = {k: [] for k in ("logit", "exp", "fp", "tmax")}
    for b in idx:
        E, Hd, M = make_batch([b]); o = [m.eval()(E, Hd, M) for m in models]
        lg = float(np.mean([x["logit"].item() for x in o])) / T
        inst = np.mean([x["inst"][0].cpu().numpy() for x in o], 0)
        ordl = np.mean([x["ord"][0].cpu().numpy() for x in o], 0)
        out["logit"].append(lg); out["exp"].append(float((coral_probs(ordl[None]) * np.arange(4)).sum()))
        out["fp"].append(float(np.mean(sig(inst) > 0.5))); out["tmax"].append(float(sig(inst).max()))
    return {k: np.array(v) for k, v in out.items()}

# ---------------- 3. GEA models, temperatures, thresholds; sanity check against cell C ----------------
GEA = {k: fold_models(v) for k, v in VARS.items()}
TEMP = {k: float(CLS[CLS.method == v].temp.iloc[0]) for k, v in VARS.items()}
THR = {k: float(CLS[CLS.method == v].thr.iloc[0]) for k, v in VARS.items()}
ALL = np.arange(N)
FROZEN = {k: predict(GEA[k], ALL, TEMP[k]) for k in VARS}
ref = pd.read_csv(TAB / f"external_{NAME}_per_item.csv"); ref = ref[ref["mode"] == "frozen"].set_index("item_id")
common = [i for i in man.item_id if i in ref.index]; pos = [man.item_id.tolist().index(i) for i in common]
dif = np.abs(FROZEN["SANC-MIL"]["logit"][pos] - ref.loc[common, "logit::SANC-MIL"].values)
print(f"sanity check vs cell C: {len(common)} slides, median |Δlogit| = {np.median(dif):.2e}, max = {dif.max():.2e} "
      f"(small differences come from fp16 GPU arithmetic)")

# ---------------- 4. fine-tuning ----------------
def site_loss(out, ysc, mask, neg):
    ys = (ysc == 3).float(); known = ysc != 2; loss = 0.0
    if known.any(): loss = loss + F.binary_cross_entropy_with_logits(out["logit"][known], ys[known])
    lv = torch.stack([(ysc > k).float() for k in range(3)], 1)
    loss = loss + CFG.W_ORD * F.binary_cross_entropy_with_logits(out["ord"], lv)
    s = out["inst"]
    if neg:
        nb = ysc <= 1
        if nb.any(): mm = mask[nb]; loss = loss + CFG.W_NEG * F.binary_cross_entropy_with_logits(s[nb][mm], torch.zeros_like(s[nb][mm]))
    pb = ysc == 3
    if pb.any():
        tk = []
        for r in torch.nonzero(pb).flatten():
            sv = s[r][mask[r]]; k = max(1, int(math.ceil(CFG.TOPK_FRAC * sv.numel()))); tk.append(sv.topk(k).values.mean())
        tk = torch.stack(tk); loss = loss + CFG.W_POS * F.binary_cross_entropy_with_logits(tk, torch.ones_like(tk))
    return loss

def finetune(m0, tr_idx, neg, seed):
    seed_everything(seed); rng = np.random.default_rng(seed); m = copy.deepcopy(m0).train()
    opt = torch.optim.AdamW(m.parameters(), lr=FT["lr"], weight_decay=FT["wd"])
    w = np.array(pd.Series(SC[tr_idx]).map(1.0 / pd.Series(SC[tr_idx]).value_counts()), dtype=float); w = w / w.sum()
    steps = int(math.ceil(len(tr_idx) / FT["bs"])); sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, FT["epochs"] * steps)
    ysc_all = torch.from_numpy(SC).to(DEVICE)
    for ep in range(FT["epochs"]):
        draw = rng.choice(tr_idx, size=steps * FT["bs"], replace=True, p=w)
        for s in range(steps):
            bp = draw[s * FT["bs"]:(s + 1) * FT["bs"]]
            E, Hd, M = make_batch(bp, rng, FT["max_tiles"], FT["drop"])
            loss = site_loss(m(E, Hd, M), ysc_all[torch.from_numpy(bp).to(DEVICE)], M, neg)
            opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); sched.step()
    return m.eval()

for rep in REPEATS:
    skf = StratifiedKFold(OUTER, shuffle=True, random_state=rep)
    for f, (tr, te) in enumerate(skf.split(ALL, SC)):
        rp = RESDIR / f"rep{rep}_fold{f}.pkl"
        if rp.exists(): continue
        ft, cal = train_test_split(tr, test_size=CAL_FRAC, stratify=SC[tr], random_state=rep * 100 + f)
        res = dict(te=te, cal=cal)
        for k, v in VARS.items():
            ms = [finetune(m0, ft, neg=VARIANTS[v].get("neg", False), seed=1000 * rep + 10 * f + j) for j, m0 in enumerate(GEA[k])]
            res[k] = dict(te=predict(ms, te, TEMP[k]), cal=predict(ms, cal, TEMP[k]))
            res[k + " (frozen)"] = dict(cal={kk: vv[cal] for kk, vv in FROZEN[k].items()})
            del ms; torch.cuda.empty_cache()
        pickle.dump(res, open(rp, "wb"))
        print(f"  repeat {rep} fold {f}: done ({(time.time() - t_start) / 60:.1f} min)", flush=True)

# ---------------- 5. analysis ----------------
def _cmp(y, s): s = np.asarray(s, float); P, Nn = s[y == 1], s[y == 0]; return (P[:, None] > Nn[None, :]) + 0.5 * (P[:, None] == Nn[None, :])
def _cnt(r, n, size): return np.stack([np.bincount(r.integers(0, n, n), minlength=n) for _ in range(size)]).astype(float)
def auc_ci(y, s, n=2000, seed=0):
    r = np.random.default_rng(seed); C = _cmp(y, s); b = np.einsum("bi,ij,bj->b", _cnt(r, C.shape[0], n), C, _cnt(r, C.shape[1], n)) / C.size
    return C.mean(), np.quantile(b, 0.025), np.quantile(b, 0.975)
def auc_paired(y, a, b_, n=2000, seed=1):
    r = np.random.default_rng(seed); Ca, Cb = _cmp(y, a), _cmp(y, b_); cp, cn = _cnt(r, Ca.shape[0], n), _cnt(r, Ca.shape[1], n)
    d = (np.einsum("bi,ij,bj->b", cp, Ca, cn) - np.einsum("bi,ij,bj->b", cp, Cb, cn)) / Ca.size
    return Ca.mean() - Cb.mean(), np.quantile(d, 0.025), np.quantile(d, 0.975), min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))
def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); adj = np.empty_like(p); run = 0.0
    for r_, i in enumerate(o): run = max(run, min(1.0, (len(p) - r_) * p[i])); adj[i] = run
    return adj
def fp_paired(a, b_, n=2000, seed=0):
    d = a - b_; r = np.random.default_rng(seed); bs = [d[r.integers(0, len(d), len(d))].mean() for _ in range(n)]
    return d.mean(), np.quantile(bs, 0.025), np.quantile(bs, 0.975), (stats.wilcoxon(a, b_).pvalue if np.any(d != 0) else 1.0), int((d < 0).sum()), int((d > 0).sum())

# per-slide out-of-fold predictions, averaged over repeats (each prediction comes from models that never saw the slide)
P = {f"{k} | adapted": {q: np.zeros(N) for q in ("logit", "exp", "fp", "tmax")} for k in VARS}
per_rep, calls = [], []
for rep in REPEATS:
    R1 = {k: {q: np.zeros(N) for q in ("logit", "exp", "fp", "tmax")} for k in VARS}
    site_call = {k: np.zeros(N, bool) for k in list(VARS) + [k + " (frozen)" for k in VARS]}
    for f in range(OUTER):
        res = pickle.load(open(RESDIR / f"rep{rep}_fold{f}.pkl", "rb")); te, cal = res["te"], res["cal"]
        for k in VARS:
            for q in R1[k]: R1[k][q][te] = res[k]["te"][q]
            for kk, pred_te in ((k, res[k]["te"]["logit"]), (k + " (frozen)", FROZEN[k]["logit"][te])):     # site threshold from the calibration set
                cl = res[kk]["cal"]["logit"]; yc = SC[cal]; ok = yc != 2
                Ts = fit_temperature(cl[ok], (yc[ok] == 3).astype(float)); thr = best_threshold((yc[ok] == 3).astype(int), sig(cl[ok] / Ts))
                site_call[kk][te] = sig(pred_te / Ts) >= thr
    st = SC != 2; y = (SC[st] == 3).astype(int)
    for k in VARS:
        for q in R1[k]: P[f"{k} | adapted"][q] += R1[k][q] / len(REPEATS)
        row = dict(repeat=rep, model=k, AUROC_logit=roc_auc_score(y, R1[k]["logit"][st]), AUROC_ordinal=roc_auc_score(y, R1[k]["exp"][st]),
                   FP_IHC0=R1[k]["fp"][SC == 0].mean(), FP_IHC1=R1[k]["fp"][SC == 1].mean(), POS_IHC3=R1[k]["fp"][SC == 3].mean(),
                   QWK=cohen_kappa_score(SC, np.clip(np.rint(R1[k]["exp"]), 0, 3).astype(int), weights="quadratic"))
        for kk, lab in ((k, "adapted"), (k + " (frozen)", "frozen")):
            src = R1[k]["logit"] if lab == "adapted" else FROZEN[k]["logit"]
            for tname, call in (("GEA threshold", sig(src) >= THR[k]), ("site threshold", site_call[kk])):
                c = call[st]; calls.append(dict(repeat=rep, model=k, weights=lab, threshold=tname,
                                                sensitivity=(c & (y == 1)).sum() / (y == 1).sum(), specificity=(~c & (y == 0)).sum() / (y == 0).sum()))
        per_rep.append(row)
PER_REP, CALLS = pd.DataFrame(per_rep), pd.DataFrame(calls)
CALLS["balanced_accuracy"] = (CALLS.sensitivity + CALLS.specificity) / 2

P.update({f"{k} | frozen": FROZEN[k] for k in VARS})
st = SC != 2; y = (SC[st] == 3).astype(int); rows = []
for k, d in P.items():
    for q, lab in (("logit", "status logit"), ("exp", "ordinal head")):
        a, lo, hi = auc_ci(y, d[q][st]); rows.append(dict(family="status_AUROC", measure=f"{k} | {lab}", value=a, lo=lo, hi=hi))
    for g, nm in ((0, "IHC 0"), (1, "IHC 1+"), (3, "IHC 3+")):
        rows.append(dict(family="tile_positive_fraction", measure=f"{k} | {nm}", value=d["fp"][SC == g].mean()))
    rows.append(dict(family="ordinal_agreement", measure=k, value=stats.spearmanr(SC, d["exp"]).correlation,
                     qwk=cohen_kappa_score(SC, np.clip(np.rint(d["exp"]), 0, 3).astype(int), weights="quadratic")))
A_, B_ = "SANC-MIL | adapted", "without constraint | adapted"
p1 = fp_paired(P[A_]["fp"][SC == 0], P[B_]["fp"][SC == 0])
p2 = auc_paired(y, P[A_]["logit"][st], P["SANC-MIL | frozen"]["logit"][st])
ph = holm([p1[3], p2[3]])
rows += [dict(family="PRIMARY", measure="P1: FP IHC 0, adapted SANC-MIL − adapted without constraint", value=p1[0], lo=p1[1], hi=p1[2], p=p1[3], p_holm=ph[0], better=p1[4], worse=p1[5]),
         dict(family="PRIMARY", measure="P2: status AUROC (logit), SANC-MIL adapted − not adapted", value=p2[0], lo=p2[1], hi=p2[2], p=p2[3], p_holm=ph[1])]
for lab, a, b_ in (("FP IHC 1+, adapted SANC-MIL − adapted without constraint", P[A_]["fp"][SC == 1], P[B_]["fp"][SC == 1]),
                   ("FP IHC 0, SANC-MIL adapted − not adapted", P[A_]["fp"][SC == 0], P["SANC-MIL | frozen"]["fp"][SC == 0])):
    r_ = fp_paired(a, b_); rows.append(dict(family="secondary_paired", measure=lab, value=r_[0], lo=r_[1], hi=r_[2], p=r_[3], better=r_[4], worse=r_[5]))
for lab, a, b_ in (("status AUROC (logit), adapted SANC-MIL − adapted without constraint", P[A_]["logit"][st], P[B_]["logit"][st]),
                   ("status AUROC (ordinal), adapted SANC-MIL − adapted without constraint", P[A_]["exp"][st], P[B_]["exp"][st])):
    r_ = auc_paired(y, a, b_); rows.append(dict(family="secondary_paired", measure=lab, value=r_[0], lo=r_[1], hi=r_[2], p=r_[3]))
SUM = pd.DataFrame(rows)
slide = man[["item_id", "ihc_score"]].copy()
for k, d in P.items():
    for q in d: slide[f"{q}::{k}"] = d[q]
atomic_csv(slide, TAB / "external_adel_site_adaptation_per_slide.csv", index=False)
atomic_csv(SUM, TAB / "external_adel_site_adaptation_summary.csv", index=False)
atomic_csv(PER_REP, TAB / "external_adel_site_adaptation_per_repeat.csv", index=False)
atomic_csv(CALLS, TAB / "external_adel_site_adaptation_thresholds.csv", index=False)
pd.set_option("display.width", 200)
print("\n", SUM.round(4).to_string(index=False))
print("\nper repeat:\n", PER_REP.round(3).to_string(index=False))
print("\noperating points (mean over repeats):\n", CALLS.groupby(["model", "weights", "threshold"])[["sensitivity", "specificity", "balanced_accuracy"]].mean().round(3).to_string())

# quick-look figure (the paper figure is redrawn from the CSV files)
fig, ax = plt.subplots(1, 3, figsize=(12, 3.4))
labs = ["SANC-MIL | frozen", "SANC-MIL | adapted", "without constraint | frozen", "without constraint | adapted"]
au = SUM[SUM.family == "status_AUROC"].set_index("measure")
for j, q in enumerate(("status logit", "ordinal head")):
    v = [au.loc[f"{l} | {q}"] for l in labs]
    ax[0].errorbar(np.arange(4) + 0.15 * (j * 2 - 1), [x.value for x in v], yerr=[[x.value - x.lo for x in v], [x.hi - x.value for x in v]], fmt="o", label=q, capsize=2)
ax[0].set_xticks(range(4)); ax[0].set_xticklabels([l.replace(" | ", "\n") for l in labs], fontsize=7); ax[0].set_ylabel("status AUROC (0/1+ vs 3+)"); ax[0].legend(fontsize=7)
tp = SUM[SUM.family == "tile_positive_fraction"].set_index("measure")
for j, g in enumerate(("IHC 0", "IHC 1+", "IHC 3+")):
    ax[1 if g != "IHC 3+" else 2].bar(np.arange(4) + (0.4 * (j % 2) - 0.2 if g != "IHC 3+" else 0), [tp.loc[f"{l} | {g}"].value for l in labs], width=0.38, label=g)
for a_ in ax[1:]: a_.set_xticks(range(4)); a_.set_xticklabels([l.replace(" | ", "\n") for l in labs], fontsize=7); a_.legend(fontsize=7)
ax[1].set_ylabel("fraction of tiles called positive (negatives)"); ax[2].set_ylabel("fraction of tiles called positive (3+)")
fig.suptitle("ADEL site adaptation (5-fold CV × 3 repeats; out-of-fold predictions)", fontsize=10); savefig(fig, "fig14_adel_site_adaptation")

zp = OUT.parent / "external_adel_site_adaptation.zip"
with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as zf:
    for p in list(TAB.glob("external_adel_site_adaptation_*")) + list(FIG.glob("fig14_adel_site_adaptation*")): zf.write(p, p.relative_to(OUT.parent))
print(f"\n[time] site adaptation: {(time.time() - t_start) / 60:.1f} min\nSEND THIS FILE: {zp}")
