# %% [markdown]
# ## 8. Multiple-instance learning: baselines, the proposed SANC-MIL, and ablations
# **SANC-MIL** stands for *Stain-Anchored, Negative-Constrained, ordinal* MIL. It combines the following:
# * **Additive instance-to-bag decomposition.** The bag logit is $z=\sum_t a_t s_t$, where $s_t$ is a per-tile *positivity* logit and $a_t$ is
#   gated attention. Each tile's contribution $a_t s_t$ is therefore exact. Attention alone only says "important", which includes important
#   *negative* evidence. That is why attention maps are not positivity maps.
# * **Negative-bag constraint.** In IHC 0/1+ cores, a pathologist saw no 3+-type staining. Under the MIL assumption, every tile there is a known
#   negative. This is the one piece of instance-level truth the labels contain. It is not applied to 2+/FISH-negative cores, which are equivocal.
# * **Top-k positive constraint.** HER2-positive cores must contain at least a small positive cluster ($k=\lceil 5\%\,n\rceil$). This mirrors the
#   biopsy rule that a cohesive cluster suffices. No proportion rule is imposed.
# * **Stain anchoring.** The 14 calibrated stain/morphology features are fused with the deep embedding.
# * **Ordinal head.** CORAL on IHC 0–3, trained jointly with the FISH-resolved status.
#
# **Baselines:** the stain-intensity rule of Pisula *et al.* (maximum tile DAB, then logistic regression), a calibrated stain-feature logistic
# regression, mean pooling, max pooling, ABMIL (Ilse 2018, the BJC 2023 method), CLAM-SB and TransMIL-lite. All of them use the same tiles,
# features, folds and class-balanced sampling.

# %%
class BagStore:
    """All tiles of all bags resident on the GPU; batches are built by index gather (fast padded batching)."""
    def __init__(self, ids, key="emb"):
        self.ids = list(ids); self.pos = {iid: k for k, iid in enumerate(self.ids)}
        embs = [BAGS[i][key] for i in self.ids]; hands = [BAGS[i]["hand_z"] for i in self.ids]
        self.n = np.array([len(e) for e in embs]); self.off = np.concatenate([[0], np.cumsum(self.n)[:-1]])
        self.E = torch.from_numpy(np.concatenate(embs)).to(DEVICE)
        self.Hd = torch.from_numpy(np.concatenate(hands)).to(DEVICE).half()
    def batch(self, bag_pos, max_tiles=None, drop=0.0, rng=None):
        sel = []
        for b in bag_pos:
            idx = np.arange(self.n[b])
            if max_tiles and len(idx) > max_tiles: idx = np.sort(rng.choice(idx, max_tiles, replace=False))
            if drop > 0 and len(idx) > 4:
                k = rng.random(len(idx)) >= drop
                idx = idx[k] if k.sum() > 0 else idx
            sel.append(idx)
        L = max(len(s) for s in sel)
        I = np.full((len(sel), L), -1, np.int64)
        for r, (b, s) in enumerate(zip(bag_pos, sel)): I[r, :len(s)] = self.off[b] + s
        It = torch.from_numpy(I).to(DEVICE); mask = It >= 0; Ic = It.clamp(min=0)
        return self.E[Ic].float(), self.Hd[Ic].float(), mask, sel

_STORES = {}
def use_embedding(key="emb"):
    """Switch the tile-embedding store (ImageNet ResNet50 'emb' or pathology foundation model 'emb2')."""
    global STORE, EMB_DIM
    if key not in _STORES: _STORES[key] = BagStore(list(TR.image_id) + list(TE.image_id), key)
    STORE = _STORES[key]; EMB_DIM = STORE.E.shape[1]
use_embedding("emb")
TRP = np.array([STORE.pos[i] for i in TR.image_id]); TEP = np.array([STORE.pos[i] for i in TE.image_id])

def masked_softmax(lg, mask):
    return torch.softmax(lg.masked_fill(~mask, -1e4), dim=1)

class GatedAttn(nn.Module):
    def __init__(self, d, da=128):
        super().__init__(); self.V = nn.Linear(d, da); self.U = nn.Linear(d, da); self.w = nn.Linear(da, 1)
    def forward(self, h): return self.w(torch.tanh(self.V(h)) * torch.sigmoid(self.U(h))).squeeze(-1)

class MILModel(nn.Module):
    def __init__(self, arch, use_emb=True, use_hand=False, pool="additive", ordinal=False, d=256, p=0.25):
        super().__init__()
        self.arch, self.use_emb, self.use_hand, self.pool, self.ordinal = arch, use_emb, use_hand, pool, ordinal
        din = (EMB_DIM if use_emb else 0) + (len(HAND_NAMES) if use_hand else 0)
        self.hand_proj = nn.Sequential(nn.Linear(len(HAND_NAMES), 64), nn.ReLU()) if use_hand else None
        din = (EMB_DIM if use_emb else 0) + (64 if use_hand else 0)
        self.proj = nn.Sequential(nn.Linear(din, d), nn.ReLU(), nn.Dropout(p))
        self.cls = nn.Linear(d, 1)
        if arch in ("abmil", "clam", "sanc"): self.attn = GatedAttn(d)
        if arch in ("max", "sanc"): self.inst = nn.Sequential(nn.Dropout(p), nn.Linear(d, 1))
        if arch == "clam": self.inst_clf = nn.ModuleList([nn.Linear(d, 2) for _ in range(2)])
        if arch == "transmil":
            self.cls_tok = nn.Parameter(torch.zeros(1, 1, d))
            self.tf = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, 4, 2 * d, 0.1, batch_first=True), 2)
        if ordinal:
            self.ord_fc = nn.Linear(d, 1, bias=False); self.ord_b = nn.Parameter(torch.tensor([0.0, -1.0, -2.0]))

    def forward(self, emb, hand, mask):
        parts = []
        if self.use_emb: parts.append(emb)
        if self.use_hand: parts.append(self.hand_proj(hand))
        h = self.proj(torch.cat(parts, -1)); m = mask.float()
        out = {"inst": None, "attn": None}
        if self.arch == "mean":
            z = (h * m[..., None]).sum(1) / m.sum(1, keepdim=True).clamp(min=1)
            out["logit"] = self.cls(z).squeeze(-1); out["inst"] = self.cls(h).squeeze(-1)
        elif self.arch == "max":
            s = self.inst(h).squeeze(-1); out["inst"] = s
            out["logit"] = s.masked_fill(~mask, -1e4).max(1).values
            z = (h * m[..., None]).sum(1) / m.sum(1, keepdim=True).clamp(min=1)
        elif self.arch in ("abmil", "clam"):
            a = masked_softmax(self.attn(h), mask); out["attn"] = a
            z = (a[..., None] * h).sum(1); out["logit"] = self.cls(z).squeeze(-1)
            if self.arch == "clam":
                l2 = self.inst_clf[1](h); out["inst"] = l2[..., 1] - l2[..., 0]; out["h"] = h
        elif self.arch == "transmil":
            t = torch.cat([self.cls_tok.expand(h.shape[0], -1, -1), h], 1)
            km = torch.cat([torch.zeros(h.shape[0], 1, dtype=torch.bool, device=h.device), ~mask], 1)
            z = self.tf(t, src_key_padding_mask=km)[:, 0]; out["logit"] = self.cls(z).squeeze(-1)
        elif self.arch == "sanc":
            a = masked_softmax(self.attn(h), mask); s = self.inst(h).squeeze(-1)
            out["attn"], out["inst"] = a, s
            z = (a[..., None] * h).sum(1)
            out["logit"] = (a * s).sum(1) if self.pool == "additive" else self.cls(z).squeeze(-1)
            out["contrib"] = a * s
        if self.ordinal:
            out["ord"] = self.ord_fc(z) + self.ord_b
        return out

VARIANTS = {
    "MeanPool":                     dict(arch="mean"),
    "MaxPool":                      dict(arch="max"),
    "ABMIL (Ilse; BJC-2023)":       dict(arch="abmil"),
    "CLAM-SB":                      dict(arch="clam"),
    "TransMIL-lite":                dict(arch="transmil"),
    "SANC-MIL (proposed)":          dict(arch="sanc", use_hand=True, neg=True, pos=True, ordinal=True, pool="additive"),
    "SANC − stain features":        dict(arch="sanc", use_hand=False, neg=True, pos=True, ordinal=True, pool="additive"),
    "SANC − negative constraint":   dict(arch="sanc", use_hand=True, neg=False, pos=True, ordinal=True, pool="additive"),
    "SANC − top-k constraint":      dict(arch="sanc", use_hand=True, neg=True, pos=False, ordinal=True, pool="additive"),
    "SANC − ordinal head":          dict(arch="sanc", use_hand=True, neg=True, pos=True, ordinal=False, pool="additive"),
    "SANC − additive (attn pool)":  dict(arch="sanc", use_hand=True, neg=True, pos=True, ordinal=True, pool="attention"),
    "Stain-only MIL (no deep emb)": dict(arch="sanc", use_emb=False, use_hand=True, neg=True, pos=True, ordinal=True, pool="additive"),
}
PROPOSED = "SANC-MIL (proposed)"; MAIN_BASELINE = "ABMIL (Ilse; BJC-2023)"

def build(v):
    return MILModel(v["arch"], use_emb=v.get("use_emb", True), use_hand=v.get("use_hand", False),
                    pool=v.get("pool", "additive"), ordinal=v.get("ordinal", False)).to(DEVICE)

Y_STATUS = torch.tensor(np.concatenate([TR.her2_status.values, TE.her2_status.values]), dtype=torch.float32, device=DEVICE)
Y_SCORE = torch.tensor(np.concatenate([TR.ihc_score.values, TE.ihc_score.values]), dtype=torch.long, device=DEVICE)

def mil_loss(out, bpos, v, mask):
    ys, ysc = Y_STATUS[bpos], Y_SCORE[bpos]
    loss = F.binary_cross_entropy_with_logits(out["logit"], ys)
    if v.get("ordinal"):
        lv = torch.stack([(ysc > k).float() for k in range(3)], 1)
        loss = loss + CFG.W_ORD * F.binary_cross_entropy_with_logits(out["ord"], lv)
    s = out["inst"]
    if v.get("neg") and s is not None:
        nb = (ysc <= 1) & (ys == 0)
        if nb.any():
            mm = mask[nb]
            loss = loss + CFG.W_NEG * F.binary_cross_entropy_with_logits(s[nb][mm], torch.zeros_like(s[nb][mm]))
    if v.get("pos") and s is not None:
        pb = ys == 1
        if pb.any():
            tk = []
            for r in torch.nonzero(pb).flatten():
                sv = s[r][mask[r]]; k = max(1, int(math.ceil(CFG.TOPK_FRAC * sv.numel())))
                tk.append(sv.topk(k).values.mean())
            tk = torch.stack(tk); loss = loss + CFG.W_POS * F.binary_cross_entropy_with_logits(tk, torch.ones_like(tk))
    if v["arch"] == "clam":
        a, h = out["attn"], out["h"]; il = []
        for r in range(h.shape[0]):
            n = int(mask[r].sum()); k = max(1, min(8, n // 4))
            if n < 2: continue
            ar = a[r][:n]; top = ar.topk(k).indices; bot = (-ar).topk(k).indices
            lg = model_clam_inst(h[r], int(ys[r].item()), top, bot)
            il.append(lg)
        if il: loss = loss + CFG.W_CLAM * torch.stack(il).mean()
    return loss

_CLAM_REF = {}
def model_clam_inst(hr, y, top, bot):
    m = _CLAM_REF["m"]; clf = m.inst_clf[y]
    lg = clf(torch.cat([hr[top], hr[bot]]))
    tgt = torch.cat([torch.ones(len(top)), torch.zeros(len(bot))]).long().to(hr.device)
    return F.cross_entropy(lg, tgt)

def train_mil(v, train_pos, seed, epochs=None):
    seed_everything(seed); rng = np.random.default_rng(seed)
    m = build(v); _CLAM_REF["m"] = m
    opt = torch.optim.AdamW(m.parameters(), lr=CFG.MIL_LR, weight_decay=CFG.MIL_WD)
    epochs = epochs or CFG.MIL_EPOCHS
    grp = np.concatenate([TR.group.values, TE.group.values])[train_pos]
    freq = pd.Series(grp).value_counts(); w = np.array([1.0 / freq[g] for g in grp]); w /= w.sum()
    steps = int(math.ceil(len(train_pos) / CFG.MIL_BS))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=CFG.MIL_LR, total_steps=epochs * steps, pct_start=0.1)
    m.train(); m.loss_hist = []
    for ep in range(epochs):
        draw = rng.choice(train_pos, size=steps * CFG.MIL_BS, replace=True, p=w); ep_loss = []
        for s in range(steps):
            bp = draw[s * CFG.MIL_BS:(s + 1) * CFG.MIL_BS]
            e, hd, mask, _ = STORE.batch(bp, CFG.MIL_MAX_TILES_TRAIN, CFG.MIL_TILE_DROP, rng)
            out = m(e, hd, mask); loss = mil_loss(out, torch.from_numpy(bp).to(DEVICE), v, mask)
            opt.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); sched.step(); ep_loss.append(loss.item())
        m.loss_hist.append(float(np.mean(ep_loss)))
    m.eval(); return m

def _enable_dropout(m):
    for mod in m.modules():
        if isinstance(mod, nn.Dropout): mod.train()

@torch.no_grad()
def predict_mil(m, pos, mc=0, bs=8):
    m.eval(); res = []
    for b in range(0, len(pos), bs):
        bp = pos[b:b + bs]
        e, hd, mask, sel = STORE.batch(bp)
        out = m(e, hd, mask)
        mcl = None
        if mc:
            _enable_dropout(m); mcl = torch.stack([m(e, hd, mask)["logit"] for _ in range(mc)], 1).cpu().numpy(); m.eval()
        for r in range(len(bp)):
            n = int(mask[r].sum())
            d = dict(logit=float(out["logit"][r]), inst=None if out["inst"] is None else out["inst"][r, :n].cpu().numpy(),
                     attn=None if out["attn"] is None else out["attn"][r, :n].cpu().numpy())
            if "ord" in out: d["ord"] = out["ord"][r].cpu().numpy()
            if mcl is not None: d["mc"] = mcl[r]
            res.append(d)
    return res

def slug(name): return re.sub("[^A-Za-z0-9]+", "_", name).strip("_")
def model_path(name, k): return OUT / "models" / f"{slug(name)}_fold{k}.pt"
def fold_models(name):
    """Fold models of `name` (loaded if saved, otherwise re-trained deterministically and saved)."""
    v = VARIANTS[name]; ms = []
    for k in range(CFG.N_FOLDS):
        p = model_path(name, k)
        if p.exists():
            m = build(v); m.load_state_dict(torch.load(p, map_location=DEVICE))
        else:
            m = train_mil(v, TRP[TR.fold.values != k], seed=k); atomic_torch_save(m.state_dict(), p)
        ms.append(m.eval())
    return ms

def run_cv(name, v, fold_col="fold", seed=0, mc_test=0, emb="emb"):
    tag = "" if emb == "emb" else f"_{emb}"
    path = CACHE / "mil" / f"{re.sub('[^A-Za-z0-9]+', '_', name)}_{fold_col}_s{seed}{tag}.pkl"
    cached = load_pickle(path) if path.exists() else None
    if cached is not None: return cached
    use_embedding(emb)
    oof = [None] * len(TR); test_runs = []; losses = []
    for k in range(CFG.N_FOLDS):
        tr_idx = np.where(TR[fold_col].values != k)[0]; va_idx = np.where(TR[fold_col].values == k)[0]
        m = train_mil(v, TRP[tr_idx], seed=seed * 100 + k)
        for j, d in zip(va_idx, predict_mil(m, TRP[va_idx], mc=mc_test)): oof[j] = d
        test_runs.append(predict_mil(m, TEP, mc=mc_test)); losses.append(getattr(m, "loss_hist", []))
        if name in (PROPOSED, MAIN_BASELINE) and fold_col == "fold" and seed == 0 and emb == "emb":
            atomic_torch_save(m.state_dict(), model_path(name, k))
        del m; torch.cuda.empty_cache()
    use_embedding("emb")
    res = dict(oof=oof, test=test_runs, loss=losses); atomic_pickle(res, path); return res

MIL_RES = {}
t0 = time.time()
for name, v in VARIANTS.items():
    mc = CFG.MC_SAMPLES if name in (PROPOSED, MAIN_BASELINE) else 0
    MIL_RES[name] = run_cv(name, v, mc_test=mc)
    print(f"{name:32s} done  ({(time.time() - t0) / 60:.1f} min)")
LEAK_RES = {n: run_cv(n, VARIANTS[n], fold_col="fold_random") for n in (PROPOSED, MAIN_BASELINE)}
# repeated CV seeds for the variants the paper's claims rest on (seed 0 = the main run above)
SEED_VARIANTS = [PROPOSED, "SANC − stain features", "SANC − negative constraint", MAIN_BASELINE, "MaxPool"]
SEED_RES = {n: [MIL_RES[n]] + [run_cv(n, VARIANTS[n], seed=sd) for sd in range(1, CFG.N_SEEDS)] for n in SEED_VARIANTS}
# encoder ablation: same MIL variants on pathology-foundation-model tile embeddings (extracted in pass 1)
ENC_VARIANTS = [PROPOSED, "SANC − stain features", MAIN_BASELINE, "MaxPool"]
ENC_RES = {}
if HAS_EMB2:
    for n in ENC_VARIANTS:
        ENC_RES[f"{n} [{CFG.ENCODER2.split('/')[-1]}]"] = run_cv(n, VARIANTS[n], emb="emb2")
    print("encoder ablation done:", list(ENC_RES))
section_time("MIL training (variants, leakage, seeds, encoder ablation)")

# %% [markdown]
# ### 8b. Grad-CAM baseline: whole-core CNN (the approach of the original draft and of Pisula et al.'s ResNet)
# A ResNet-18 is trained on the whole-core thumbnail (≈1.8 µm/px) with the same folds and class-balanced sampling. Its
# Grad-CAM map is pooled onto the same 224-px tiles, so it can be scored with the **same** localisation metrics.
#
# Two read-outs are reported:
# 1. the per-image min–max-normalised CAM, as in the original draft
# 2. CAM × P(HER2+)
#
# Per-image normalisation guarantees "hot" tiles even in IHC-0 cores. The tile false-positive rate in IHC-0 cores
# quantifies that flaw.

# %%
CAM_SIZE = 672 if RUN_MODE != "debug" else 128
def _load_thumb(iid):
    im = cv2.cvtColor(cv2.imread(str(THUMB_DIR / f"{iid}.jpg")), cv2.COLOR_BGR2RGB)
    return cv2.resize(im, (CAM_SIZE, CAM_SIZE), interpolation=cv2.INTER_AREA), im.shape[:2]
def _cam_logit(v): v = np.clip(v, 1e-4, 1 - 1e-4); return np.log(v / (1 - v))

def run_gradcam_baseline():
    path = CACHE / "mil" / "gradcam_thumbnail_cnn_v2.pkl"
    cached = load_pickle(path) if path.exists() else None
    if cached is not None: return cached
    ids = list(TR.image_id) + list(TE.image_id); shapes = {}
    Xn = np.empty((len(ids), CAM_SIZE, CAM_SIZE, 3), np.uint8)       # filled in place: 1,905 x 672^2 x 3 = 2.6 GB, no duplicate copy
    for k, iid in enumerate(tqdm(ids, desc="load thumbnails")): Xn[k], shapes[iid] = _load_thumb(iid)
    X = torch.from_numpy(Xn).permute(0, 3, 1, 2)                     # uint8 view on CPU; batches go to the GPU one at a time
    ypos = np.concatenate([TR.her2_status.values, TE.her2_status.values]).astype(np.float32)
    grp = np.concatenate([TR.group.values, TE.group.values]); ntr = len(TR)
    mean = torch.tensor(IMNET_MEAN, device=DEVICE).view(1, 3, 1, 1); std = torch.tensor(IMNET_STD, device=DEVICE).view(1, 3, 1, 1)
    def mk():
        try: return timm.create_model(CFG.CAM_ARCH, pretrained=True, num_classes=1).to(DEVICE)
        except Exception: return timm.create_model(CFG.CAM_ARCH, pretrained=False, num_classes=1).to(DEVICE)
    def tile_scores(cam, iid):
        h0, w0 = shapes[iid]; c = cv2.resize(cam, (w0, h0), interpolation=cv2.INTER_LINEAR)
        T, R = CFG.TILE, CFG.THUMB_REL
        out = []
        for y, x in BAGS[iid]["coords"]:      # clamp to >=1 px inside the thumbnail, so an edge tile can never give an empty-slice NaN
            y0 = min(y // R, h0 - 1); x0 = min(x // R, w0 - 1)
            out.append(c[y0:max(y0 + 1, (y + T) // R), x0:max(x0 + 1, (x + T) // R)].mean())
        return np.array(out, np.float32)
    def gradcam(m, idx):
        feats = {}
        h = m.layer4.register_forward_hook(lambda mod, i, o: feats.__setitem__("a", o))
        out = []
        m.eval()
        for b in range(0, len(idx), 8):
            bi = idx[b:b + 8]; x = ((X[bi].to(DEVICE).float() / 255) - mean) / std
            x.requires_grad_(False); lg = m(x).squeeze(1); a = feats["a"]
            g = torch.autograd.grad(lg.sum(), a)[0]
            cam = F.relu((g.mean((2, 3), keepdim=True) * a).sum(1)).detach().cpu().numpy()
            for j, k in enumerate(bi):
                cn = np.nan_to_num(cam[j], nan=0.0, posinf=0.0, neginf=0.0); cn = (cn - cn.min()) / (cn.max() - cn.min() + 1e-8)
                ts = tile_scores(cn, ids[k]); out.append((float(lg[j].detach()), ts))
        h.remove(); return out
    oof = [None] * ntr; test_norm, test_prob = [], []
    for k in range(CFG.N_FOLDS):
        tr_idx = np.where(TR.fold.values != k)[0]; va_idx = np.where(TR.fold.values == k)[0]
        fold_path = CACHE / "mil" / f"gradcam_v2_fold{k}.pkl"; fk = load_pickle(fold_path) if fold_path.exists() else None
        if fk is not None:                       # resume: this fold finished in an earlier run
            for j, d in zip(va_idx, fk["oof"]): oof[j] = d
            test_norm.append(fk["test_norm"]); test_prob.append(fk["test_prob"])
            print(f"  Grad-CAM CNN fold {k}: done earlier, loaded from cache"); continue
        seed_everything(500 + k); m = mk(); opt = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=1e-4)
        freq = pd.Series(grp[tr_idx]).value_counts(); w = np.array([1 / freq[g] for g in grp[tr_idx]]); w /= w.sum()
        steps = int(math.ceil(len(tr_idx) / CFG.CAM_BS))
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-4, total_steps=max(20, CFG.CAM_EPOCHS * steps), pct_start=0.1)
        rng = np.random.default_rng(k)
        # fp16 autocast needs loss scaling, otherwise small gradients underflow to zero and the baseline is under-trained
        cam_scaler = torch.amp.GradScaler("cuda", enabled=DEVICE.type == "cuda") if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler") \
            else torch.cuda.amp.GradScaler(enabled=DEVICE.type == "cuda")
        for ep in range(CFG.CAM_EPOCHS):
            m.train(); draw = rng.choice(tr_idx, steps * CFG.CAM_BS, replace=True, p=w)
            for st in range(steps):
                bi = draw[st * CFG.CAM_BS:(st + 1) * CFG.CAM_BS]; x = X[bi].to(DEVICE).float() / 255
                if random.random() < 0.5: x = x.flip(-1)
                if random.random() < 0.5: x = x.flip(-2)
                x = torch.rot90(x, random.randint(0, 3), (-2, -1))
                x = (x * (1 + (torch.rand(len(bi), 3, 1, 1, device=DEVICE) - 0.5) * 0.1)).clamp(0, 1)
                with autocast(): lg = m((x - mean) / std).squeeze(1)
                loss = F.binary_cross_entropy_with_logits(lg.float(), torch.from_numpy(ypos[bi]).to(DEVICE))
                opt.zero_grad(set_to_none=True); cam_scaler.scale(loss).backward(); cam_scaler.step(opt); cam_scaler.update()
                if sched.last_epoch < sched.total_steps - 1: sched.step()
        for j, (lg, ts) in zip(va_idx, gradcam(m, list(va_idx))): oof[j] = dict(logit=lg, inst=None, attn=None)
        te = gradcam(m, list(range(ntr, len(ids))))
        test_norm.append([dict(logit=lg, inst=_cam_logit(ts), attn=None) for lg, ts in te])
        test_prob.append([dict(logit=lg, inst=_cam_logit(ts / (1 + math.exp(-lg))), attn=None) for lg, ts in te])
        atomic_pickle(dict(oof=[oof[j] for j in va_idx], test_norm=test_norm[-1], test_prob=test_prob[-1]), fold_path)
        print(f"  Grad-CAM CNN fold {k} done"); del m; torch.cuda.empty_cache()
    res = {"Grad-CAM CNN (per-image normalised CAM)": dict(oof=oof, test=test_norm),
           "Grad-CAM CNN (CAM × P(HER2+))": dict(oof=oof, test=test_prob)}
    atomic_pickle(res, path); return res

CAM_RES = run_gradcam_baseline() if CFG.RUN_GRADCAM else {}
section_time("Grad-CAM baseline")

# %%
# ---------------- non-MIL baselines (stain intensity rules) ----------------
def core_stain_descriptor(iid):
    h = BAGS[iid]["hand_raw"]; g = lambda n: h[:, HAND_NAMES.index(n)]
    top = lambda x: np.sort(x)[-max(1, int(math.ceil(0.05 * len(x)))):].mean()
    return [g("dab_mean").max(), top(g("mem_weak")), top(g("mem_mod")), top(g("mem_strong")), g("mem_weak").mean(),
            g("mem_mod").mean(), g("mem_strong").mean(), g("nonmem_dab").mean(), g("nuclear_dab").mean()]
XS_tr = np.array([core_stain_descriptor(i) for i in TR.image_id]); XS_te = np.array([core_stain_descriptor(i) for i in TE.image_id])

def sk_cv(Xtr, Xte, y, fold_col="fold"):
    oof = np.zeros(len(Xtr)); te = []
    for k in range(CFG.N_FOLDS):
        tr = TR[fold_col].values != k
        mdl = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=2000, C=1.0)).fit(Xtr[tr], y[tr])
        oof[~tr] = mdl.decision_function(Xtr[~tr]); te.append(mdl.decision_function(Xte))
    return oof, np.array(te)
ystat_tr, ystat_te = TR.her2_status.values, TE.her2_status.values
SK_RES = {"Stain intensity rule (Pisula 2023)": sk_cv(XS_tr[:, :1], XS_te[:, :1], ystat_tr),
          "Calibrated stain features + LR": sk_cv(XS_tr, XS_te, ystat_tr)}

# %% [markdown]
# ## 9. Classification evaluation (independent of localisation)
# Logits are averaged over the 5 fold models. The temperature and the decision threshold (maximum balanced accuracy) are fitted **on the OOF
# predictions only**, then frozen for the test set. CIs come from a stratified bootstrap with `N_BOOT` resamples. The paired bootstrap ΔAUC compares
# the proposed model with each baseline.
#
# With only about 16 positive test cores, CIs are wide. Small differences are **not** evidence of superiority.

# %%
def sig(x): return 1 / (1 + np.exp(-np.asarray(x, float)))
def fit_temperature(logit, y):
    lt = torch.tensor(logit, dtype=torch.float64); yt = torch.tensor(y, dtype=torch.float64)
    lT = torch.zeros(1, dtype=torch.float64, requires_grad=True); opt = torch.optim.LBFGS([lT], lr=0.1, max_iter=200)
    def cl():
        opt.zero_grad(); l = F.binary_cross_entropy_with_logits(lt / lT.exp(), yt); l.backward(); return l
    opt.step(cl); return float(lT.exp())
def best_threshold(y, p):
    grid = np.unique(np.quantile(p, np.linspace(0.01, 0.99, 199)))
    return float(grid[np.argmax([balanced_accuracy_score(y, p >= t) for t in grid])])
def ece(y, p, n=10):
    b = np.minimum((p * n).astype(int), n - 1); e = 0
    for k in range(n):
        s = b == k
        if s.any(): e += s.mean() * abs(p[s].mean() - y[s].mean())
    return float(e)
def metrics(y, p, thr):
    yh = (p >= thr).astype(int); tn, fp, fn, tp = confusion_matrix(y, yh, labels=[0, 1]).ravel()
    return dict(AUROC=roc_auc_score(y, p) if len(set(y)) > 1 else np.nan, AUPRC=average_precision_score(y, p),
                BalAcc=balanced_accuracy_score(y, yh), Sens=tp / max(tp + fn, 1), Spec=tn / max(tn + fp, 1),
                F1=f1_score(y, yh, zero_division=0), MCC=matthews_corrcoef(y, yh), Brier=brier_score_loss(y, p), ECE=ece(y, p))
def boot_ci(y, p, thr, n=None, seed=0):
    n = n or CFG.N_BOOT; r = np.random.default_rng(seed); pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]; out = []
    for _ in range(n):
        idx = np.concatenate([r.choice(pos, len(pos)), r.choice(neg, len(neg))])
        out.append(metrics(y[idx], p[idx], thr))
    d = pd.DataFrame(out); return d.quantile(0.025), d.quantile(0.975)

def summarize(name, oof_logit, test_logit_folds, y_tr=ystat_tr, y_te=ystat_te):
    T = fit_temperature(oof_logit, y_tr); p_oof = sig(oof_logit / T)
    thr = best_threshold(y_tr, p_oof)
    p_te = sig(np.mean(test_logit_folds, 0) / T)
    m_te = metrics(y_te, p_te, thr); lo, hi = boot_ci(y_te, p_te, thr)
    m_oof = metrics(y_tr, p_oof, thr)
    row = {"method": name, "temp": T, "thr": thr, "OOF_AUROC": m_oof["AUROC"], "OOF_BalAcc": m_oof["BalAcc"]}
    for k in ["AUROC", "AUPRC", "BalAcc", "Sens", "Spec", "F1", "MCC", "Brier", "ECE"]:
        row[k] = m_te[k]; row[k + "_lo"] = lo[k]; row[k + "_hi"] = hi[k]
    return row, p_oof, p_te

ROWS, P_TE, P_OOF = [], {}, {}
for name, (oof, te) in SK_RES.items():
    r, po, pt = summarize(name, oof, te); ROWS.append(r); P_TE[name] = pt; P_OOF[name] = po
_cls_sources = {**MIL_RES, **ENC_RES}
if CAM_RES: _cls_sources["Grad-CAM CNN (thumbnail)"] = next(iter(CAM_RES.values()))
for name, res in _cls_sources.items():
    oof = np.array([d["logit"] for d in res["oof"]]); te = np.array([[d["logit"] for d in run] for run in res["test"]])
    r, po, pt = summarize(name, oof, te); ROWS.append(r); P_TE[name] = pt; P_OOF[name] = po
CLS = pd.DataFrame(ROWS)
fmt = lambda r, k: f"{r[k]:.3f} [{r[k + '_lo']:.3f}–{r[k + '_hi']:.3f}]"
CLS_PRETTY = pd.DataFrame({"method": CLS.method, "OOF AUROC (grouped CV)": CLS.OOF_AUROC.round(3),
                           **{f"Test {k}": [fmt(r, k) for _, r in CLS.iterrows()] for k in ["AUROC", "AUPRC", "BalAcc", "Sens", "Spec", "MCC"]},
                           "Test ECE": CLS.ECE.round(3)})
CLS.to_csv(TAB / "classification_full.csv", index=False); CLS_PRETTY.to_csv(TAB / "classification_table.csv", index=False)
print(CLS_PRETTY.to_string(index=False))

def paired_delta_auc(pa, pb, y, n=None, seed=1):
    n = n or CFG.N_BOOT; r = np.random.default_rng(seed); pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]; d = []
    for _ in range(n):
        idx = np.concatenate([r.choice(pos, len(pos)), r.choice(neg, len(neg))])
        d.append(roc_auc_score(y[idx], pa[idx]) - roc_auc_score(y[idx], pb[idx]))
    d = np.array(d); return float(np.mean(d)), float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975)), float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())))
DELTA = pd.DataFrame([dict(comparison=f"{PROPOSED} − {b}", **dict(zip(["dAUC", "lo", "hi", "p_boot"], paired_delta_auc(P_TE[PROPOSED], P_TE[b], ystat_te))))
                      for b in P_TE if b != PROPOSED])
def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); adj = np.empty_like(p); run = 0.0
    for r, i in enumerate(o): run = max(run, min(1.0, (len(p) - r) * p[i])); adj[i] = run
    return adj
DELTA["p_holm"] = holm(DELTA.p_boot.values)     # family-wise correction over all comparisons with the proposed model
DELTA.to_csv(TAB / "paired_delta_auc.csv", index=False); print(DELTA.round(3).to_string(index=False))

# per-IHC-group error profile (mirrors Table 2 of Pisula et al.)
def group_profile(name):
    r = CLS[CLS.method == name].iloc[0]; yh = (P_TE[name] >= r.thr).astype(int)
    d = TE.assign(pred=yh)
    return d.groupby("group").apply(lambda s: pd.Series({"n": len(s), "pred_pos_rate": s.pred.mean(),
                                                          "error_rate": (s.pred != s.her2_status).mean()})).reindex(GROUP_ORDER)
GP = pd.concat({n: group_profile(n) for n in [PROPOSED, MAIN_BASELINE, "Stain intensity rule (Pisula 2023)"]}, axis=1)
GP.to_csv(TAB / "test_error_by_ihc_group.csv"); print(GP.round(3))

# leakage sensitivity: grouped vs random CV (OOF AUROC). A large gap = within-patient correlation inflating random CV.
LEAK = []
for n, res in LEAK_RES.items():
    oof_r = np.array([d["logit"] for d in res["oof"]]); oof_g = np.array([d["logit"] for d in MIL_RES[n]["oof"]])
    LEAK.append(dict(method=n, OOF_AUROC_random_CV=roc_auc_score(ystat_tr, oof_r), OOF_AUROC_proxy_grouped_CV=roc_auc_score(ystat_tr, oof_g)))
LEAK = pd.DataFrame(LEAK); LEAK.to_csv(TAB / "leakage_sensitivity.csv", index=False); print(LEAK.round(4))

# QC sensitivity: test metrics without QC-flagged cores
flag_te = TE.qc_flag.values.astype(bool)
QCS = pd.DataFrame([dict(method=n, AUROC_all=roc_auc_score(ystat_te, P_TE[n]),
                         AUROC_unflagged=roc_auc_score(ystat_te[~flag_te], P_TE[n][~flag_te]) if len(set(ystat_te[~flag_te])) > 1 else np.nan,
                         n_flagged=int(flag_te.sum())) for n in [PROPOSED, MAIN_BASELINE]])
QCS.to_csv(TAB / "qc_sensitivity.csv", index=False)

# ordinal IHC score from SANC (CORAL -> class probabilities)
def coral_probs(ordl):
    pk = sig(ordl); p = np.stack([1 - pk[:, 0], pk[:, 0] - pk[:, 1], pk[:, 1] - pk[:, 2], pk[:, 2]], 1)
    return np.clip(p, 0, 1) / np.clip(p, 0, 1).sum(1, keepdims=True)
ORD_TE = coral_probs(np.mean([[d["ord"] for d in run] for run in MIL_RES[PROPOSED]["test"]], 0))
ORD_OOF = coral_probs(np.array([d["ord"] for d in MIL_RES[PROPOSED]["oof"]]))
score_pred = ORD_TE.argmax(1)
ord_metrics = dict(test_QWK=cohen_kappa_score(TE.ihc_score, score_pred, weights="quadratic"),
                   test_BalAcc_present_classes=balanced_accuracy_score(TE.ihc_score, score_pred),
                   OOF_QWK=cohen_kappa_score(TR.ihc_score, ORD_OOF.argmax(1), weights="quadratic"),
                   classes_present_in_test=sorted(TE.ihc_score.unique().tolist()))
save_json(ord_metrics, TAB / "ordinal_score_metrics.json"); print(ord_metrics)

# figures: ROC + PR + reliability + score confusion
fig, ax = plt.subplots(1, 4, figsize=(15, 3.6))
show = ["Stain intensity rule (Pisula 2023)", "Calibrated stain features + LR", MAIN_BASELINE, "CLAM-SB", PROPOSED]
for c, n in zip([PAL[3], PAL[2], PAL[0], PAL[6], PAL[1]], show):
    f, t, _ = roc_curve(ystat_te, P_TE[n]); r = CLS[CLS.method == n].iloc[0]
    ax[0].plot(f, t, color=c, label=f"{n.split(' (')[0]}  {r.AUROC:.3f} [{r.AUROC_lo:.2f}–{r.AUROC_hi:.2f}]")
    pr, rc, _ = precision_recall_curve(ystat_te, P_TE[n]); ax[1].plot(rc, pr, color=c)
ax[0].plot([0, 1], [0, 1], color="#c3c2b7", lw=1, ls="--"); ax[0].set_xlabel("1 − specificity"); ax[0].set_ylabel("sensitivity")
ax[0].legend(fontsize=6.5, loc="lower right"); ax[0].set_title(f"Test ROC (n={len(TE)}, {int(ystat_te.sum())} HER2+)")
ax[1].set_xlabel("recall"); ax[1].set_ylabel("precision"); ax[1].set_title("Test precision–recall")
for c, n in zip([PAL[0], PAL[1]], [MAIN_BASELINE, PROPOSED]):
    p = P_OOF[n]; b = np.minimum((p * 10).astype(int), 9)
    xs = [p[b == k].mean() for k in range(10) if (b == k).sum() > 5]; ys = [ystat_tr[b == k].mean() for k in range(10) if (b == k).sum() > 5]
    ax[2].plot(xs, ys, "o-", color=c, ms=4, label=n.split(" (")[0])
ax[2].plot([0, 1], [0, 1], color="#c3c2b7", lw=1, ls="--"); ax[2].set_xlabel("predicted P(HER2+)"); ax[2].set_ylabel("observed")
ax[2].set_title("Reliability (OOF, temperature-scaled)"); ax[2].legend(fontsize=7)
cm = confusion_matrix(TE.ihc_score, score_pred, labels=[0, 1, 2, 3])
ax[3].imshow(cm, cmap="Blues"); ax[3].grid(False)
for i in range(4):
    for j in range(4): ax[3].text(j, i, cm[i, j], ha="center", va="center", fontsize=8, color="#0b0b0b")
ax[3].set_xticks(range(4)); ax[3].set_yticks(range(4)); ax[3].set_xlabel("predicted IHC"); ax[3].set_ylabel("true IHC")
ax[3].set_title(f"SANC ordinal head (QWK {ord_metrics['test_QWK']:.2f})")
savefig(fig, "fig06_classification")

# %% [markdown]
# ## 10. Uncertainty, abstention, and guideline-faithful handling of equivocal cases
# * **Uncertainty.** The deep ensemble (5 folds) is combined with MC-dropout. The outputs are predictive entropy (total uncertainty) and mutual
#   information (epistemic uncertainty).
# * **Abstention.** The entropy threshold is chosen on OOF for `ABSTAIN_COVERAGE`, then applied to the test set. Risk–coverage curves show the trade-off.
# * **Equivocal routing.** If the ordinal head gives P(IHC 2) ≥ 0.3 or predicts 2, the core is flagged "**ISH required**", whatever the status
#   probability. The model's status output for those cores is reported as research information only.

# %%
def mc_stats(res, T):
    runs = np.stack([np.stack([d["mc"] for d in run]) for run in res["test"]], 1)  # (n, folds, mc)
    p = sig(runs / T).reshape(len(TE), -1); pm = p.mean(1)
    H = lambda q: -(q * np.log(q + 1e-9) + (1 - q) * np.log(1 - q + 1e-9))
    return pm, H(pm), H(pm) - H(p).mean(1)
def oof_entropy(res, T):
    p = sig(np.array([d["mc"] for d in res["oof"]]) / T); pm = p.mean(1)
    return -(pm * np.log(pm + 1e-9) + (1 - pm) * np.log(1 - pm + 1e-9))
UNC = {}
for n in [PROPOSED, MAIN_BASELINE]:
    T = CLS[CLS.method == n].temp.iloc[0]; thr = CLS[CLS.method == n].thr.iloc[0]
    pm, ent, mi = mc_stats(MIL_RES[n], T)
    tau = np.quantile(oof_entropy(MIL_RES[n], T), CFG.ABSTAIN_COVERAGE)
    keep = ent <= tau
    cov = np.linspace(0.3, 1, 15); risk = []
    order = np.argsort(ent)
    for c in cov:
        s = order[:max(2, int(c * len(order)))]; yh = (pm[s] >= thr).astype(int)
        risk.append(1 - balanced_accuracy_score(ystat_te[s], yh) if len(set(ystat_te[s])) > 1 else np.nan)
    UNC[n] = dict(pm=pm, ent=ent, mi=mi, tau=tau, keep=keep, cov=cov, risk=np.array(risk), thr=thr)
pu = UNC[PROPOSED]
# ISH routing: threshold on P(IHC 2) chosen on OUT-OF-FOLD predictions to reach the target sensitivity for IHC 2 cores
# (v1 used a fixed 0.3 cut-off and caught only 40% of test IHC 2 cores).
_oof2 = ORD_OOF[TR.ihc_score.values == 2, 2]
ISH_TAU = float(np.quantile(_oof2, 1 - CFG.ISH_ROUTE_TARGET_SENS)) if len(_oof2) else 0.5
route_oof = ORD_OOF[:, 2] >= ISH_TAU
route_ish = ORD_TE[:, 2] >= ISH_TAU
ab = dict(abstain_rate_test=float(1 - pu["keep"].mean()),
          balacc_all=float(balanced_accuracy_score(ystat_te, pu["pm"] >= pu["thr"])),
          balacc_retained=float(balanced_accuracy_score(ystat_te[pu["keep"]], pu["pm"][pu["keep"]] >= pu["thr"])) if len(set(ystat_te[pu["keep"]])) > 1 else np.nan,
          ISH_routing_sensitivity_for_true_IHC2=float(route_ish[TE.ihc_score.values == 2].mean()) if (TE.ihc_score == 2).any() else np.nan,
          ISH_routing_rate_in_non_IHC2=float(route_ish[TE.ihc_score.values != 2].mean()),
          ISH_threshold_P_IHC2=ISH_TAU, ISH_routing_OOF_sensitivity=float(route_oof[TR.ihc_score.values == 2].mean()),
          ISH_routing_OOF_rate_non_IHC2=float(route_oof[TR.ihc_score.values != 2].mean()),
          ISH_routing_test_rate_IHC0=float(route_ish[TE.ihc_score.values == 0].mean()),
          ISH_routing_test_rate_IHC3=float(route_ish[TE.ihc_score.values == 3].mean()) if (TE.ihc_score == 3).any() else np.nan,
          AURC_proposed=float(np.nanmean(UNC[PROPOSED]["risk"])), AURC_abmil=float(np.nanmean(UNC[MAIN_BASELINE]["risk"])))
save_json(ab, TAB / "abstention.json"); print(ab)
section_time("classification evaluation + uncertainty")
TE_PRED = TE[["image_id", "filename", "ihc_score", "her2_status", "group", "qc_flag"]].assign(
    p_her2=pu["pm"], entropy=pu["ent"], mutual_info=pu["mi"], abstain=~pu["keep"],
    **{f"p_ihc{k}": ORD_TE[:, k] for k in range(4)}, route_ISH=route_ish,
    decision=np.where(~pu["keep"] | TE.qc_flag.values, "ABSTAIN (refer)", np.where(route_ish, "EQUIVOCAL: ISH required",
                      np.where(pu["pm"] >= pu["thr"], "HER2-positive (model)", "HER2-negative (model)"))))
TE_PRED.to_csv(TAB / "test_predictions.csv", index=False)
fig, ax = plt.subplots(1, 2, figsize=(9, 3.3))
for c, n in zip([PAL[0], PAL[1]], [MAIN_BASELINE, PROPOSED]):
    ax[0].plot(UNC[n]["cov"], UNC[n]["risk"], "o-", ms=3, color=c, label=n.split(" (")[0])
ax[0].set_xlabel("coverage"); ax[0].set_ylabel("balanced error"); ax[0].set_title("Risk–coverage (test)"); ax[0].legend(fontsize=7)
for g in GROUP_ORDER:
    s = TE.group.values == g
    if s.any(): ax[1].scatter(pu["pm"][s], pu["ent"][s], s=14, color=GROUP_COL[g], label=f"IHC {g} (n={s.sum()})", alpha=0.8)
ax[1].axhline(pu["tau"], color="#52514e", ls="--", lw=1); ax[1].set_xlabel("P(HER2+)"); ax[1].set_ylabel("predictive entropy")
ax[1].set_title("Uncertainty by IHC group (test)"); ax[1].legend(fontsize=6.5)
savefig(fig, "fig07_uncertainty")
