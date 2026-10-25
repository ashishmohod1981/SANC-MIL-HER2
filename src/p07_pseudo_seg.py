# %% [markdown]
# ## 12. Pseudo-masks with explicit ignore regions and confidence (train cores only)
# There are two **separate** targets, so tumour or evidence localisation is never conflated with membrane morphology.
#
# **Membrane target**, driven by stain physics and morphology:
#
# | Pixel condition | Label | Confidence |
# |---|---|---|
# | Membrane-shaped (line-like, locally prominent), non-nuclear, DAB ≥ `t_weak` | 1 | 0.5–1.0, rising with DAB OD |
# | 1-px rim around each membrane (ambiguous width) | **ignore** | – |
# | DAB < `t_bg`, or background | 0 | 1.0 |
# | Non-membranous (cytoplasmic/blob) or nuclear DAB | 0 | 0.5 |
# | Faint non-linear stain | 0 | 0.5 |
# | Rim band, sub-threshold ridges, or ambiguous DAB | **ignore** (255) | – |
#
# **Evidence target**, driven by the image labels via **out-of-fold** SANC tile positivity (each core is scored by the model that never saw it):
#
# | Core group | Tiles | Label | Confidence |
# |---|---|---|---|
# | IHC 0/1+ | all | 0 | 1.0 (exact negative bag) |
# | HER2+ | P ≥ 0.8 | 1 | P |
# | HER2+ | P ≤ 0.2 | 0 | 0.5 |
# | HER2+ | otherwise | ignore | – |
# | 2+/FISH− | P ≤ 0.2 | 0 | 0.5 |
# | 2+/FISH− | otherwise | ignore | – |
#
# These are training signals, **not annotations**. They are never used as ground truth for accuracy.

# %%
CROP_DIR = CACHE / "seg_crops"
CROP_INDEX = CACHE / "seg_crops_index.csv"
OOF_INST = {TR.image_id[j]: sig(d["inst"]) for j, d in enumerate(MIL_RES[PROPOSED]["oof"])}

@torch.no_grad()
def pseudo_labels(mp, iid, row):
    T = CFG.TILE; tis = mp["tissue"][0, 0]; rim = mp["rim"][0, 0]; dab = mp["DAB"][0, 0]
    H, W = tis.shape
    lab = torch.full((H, W), 255, dtype=torch.uint8, device=DEVICE); conf = torch.zeros((H, W), dtype=torch.uint8, device=DEVICE)
    def put(m, l, c):
        lab[m] = l; conf[m] = c
    put(~tis, 0, 255)
    put(tis & (dab < CALIB["t_bg"]), 0, 255)
    put(tis & ~mp["mem_shape"][0, 0] & (dab >= CALIB["t_bg"]) & (dab < CALIB["t_weak"]), 0, 128)
    put(tis & (dab >= CALIB["t_weak"]) & (mp["nonmem"][0, 0] | mp["nuc"][0, 0]), 0, 128)   # cytoplasmic / nuclear DAB
    pos = mp["mem_weak"][0, 0]
    ring = dilate(pos[None, None], 1)[0, 0] & ~pos                     # 1-px rim of each membrane: ambiguous width -> ignore
    lab[ring & tis] = 255; conf[ring & tis] = 0
    c = 0.5 + 0.5 * ((torch.log(dab.clamp(min=1e-6)) - math.log(CALIB["t_weak"])) / (math.log(CALIB["t_strong"]) - math.log(CALIB["t_weak"]))).clamp(0, 1)
    lab[pos] = 1; conf[pos] = (c[pos] * 255).to(torch.uint8)
    lab[rim] = 255; conf[rim] = 0
    # evidence target
    elab = torch.full((H, W), 255, dtype=torch.uint8, device=DEVICE); econf = torch.zeros((H, W), dtype=torch.uint8, device=DEVICE)
    elab[~tis] = 0; econf[~tis] = 76
    p = OOF_INST.get(iid); coords = BAGS[iid]["coords"]
    neg_bag = row.ihc_score <= 1 and row.her2_status == 0
    for (y, x), pv in zip(coords, p if p is not None else np.full(len(coords), 0.5)):
        sl = (slice(y, y + T), slice(x, x + T))
        if neg_bag: l, cc = 0, 255
        elif row.her2_status == 1 and pv >= CFG.EV_P_POS: l, cc = 1, int(255 * pv)
        elif pv <= CFG.EV_P_NEG: l, cc = 0, 128
        else: l, cc = 255, 0
        elab[sl] = l; econf[sl] = cc
    elab[~tis & (elab == 255)] = 0
    return torch.stack([lab, conf, elab, econf], -1)

def sample_centers(mp, lab, K, rng):
    S = CFG.SEG_CROP; tis = mp["tissue"][0, 0]; H, W = tis.shape
    def pick(mask, n):
        idx = torch.nonzero(mask)
        if len(idx) == 0 or n <= 0: return []
        sel = idx[torch.from_numpy(rng.choice(len(idx), min(n, len(idx)), replace=False)).to(idx.device)].cpu().numpy()
        return [tuple(v) for v in sel]
    k_mem, k_ev = K // 2, K // 4
    cs = pick(lab[..., 0] == 1, k_mem) + pick(lab[..., 2] == 1, k_ev)
    cs += pick(tis & ~mp["rim"][0, 0], K - len(cs))
    return [(int(np.clip(y - S // 2, 0, H - S)), int(np.clip(x - S // 2, 0, W - S))) for y, x in cs]

def _img_seed(iid): return int(hashlib.md5(iid.encode()).hexdigest()[:8], 16) + CFG.SEED   # per-core RNG: order-independent, resumable

if CROP_INDEX.exists():
    CROPS = pd.read_csv(CROP_INDEX)
else:
    plog = ProgressLog("seg_crops")                       # one line per finished core -> restart skips finished cores
    todo_tr = TR[[i not in plog for i in TR.image_id]].reset_index(drop=True)
    print(f"pseudo-label crops: {len(plog)} cores done earlier, {len(todo_tr)} to go")
    if len(todo_tr):
        dl = torch.utils.data.DataLoader(DecodeDS(todo_tr), batch_size=1, num_workers=CFG.NUM_WORKERS, collate_fn=lambda b: b[0])
        for i, work in tqdm(dl, total=len(todo_tr), desc="pseudo-label crops"):
            r = todo_tr.iloc[i]; iid = r.image_id; S = CFG.SEG_CROP; recs = []
            if work is not None and work.shape[0] >= S and work.shape[1] >= S:
                mz = np.load(MAP_DIR / f"{iid}.npz")
                mp = ENGINE.maps(work, mz["I0"], mz["tissue"], mz["rim"])
                lab = pseudo_labels(mp, iid, r)
                for k, (y, x) in enumerate(sample_centers(mp, lab, CFG.CROPS_PER_IMAGE, np.random.default_rng(_img_seed(iid)))):
                    cp = CROP_DIR / f"{iid}_{k}.jpg"; lp = CROP_DIR / f"{iid}_{k}.png"
                    cv2.imwrite(str(cp), cv2.cvtColor(work[y:y + S, x:x + S], cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
                    cv2.imwrite(str(lp), lab[y:y + S, x:x + S].cpu().numpy())
                    recs.append(dict(image_id=iid, crop=str(cp), label=str(lp), y=int(y), x=int(x), fold=int(r.fold), group=r.group))
                del mp, lab
            plog.add(iid, dict(crops=recs))                   # logged only after all of this core's crops are on disk
    CROPS = pd.DataFrame([c for rec in plog.records() for c in rec["crops"]])
    atomic_csv(CROPS, CROP_INDEX, index=False)
print(f"{len(CROPS)} pseudo-labelled crops from {CROPS.image_id.nunique()} train cores")

def pseudo_label_figure(n=4):
    s = CROPS.groupby("group").head(1).head(n)
    fig, axes = plt.subplots(len(s), 4, figsize=(9.5, 2.4 * len(s))); axes = np.atleast_2d(axes)
    for ax, (_, r) in zip(axes, s.iterrows()):
        img = cv2.cvtColor(cv2.imread(r.crop), cv2.COLOR_BGR2RGB); lab = cv2.imread(r.label, cv2.IMREAD_UNCHANGED)
        def show_lab(a, l, c, title):
            v = np.zeros(l.shape + (3,), np.uint8) + 255
            v[l == 0] = (230, 230, 230); v[l == 1] = (227, 73, 72); v[l == 255] = (237, 161, 0)
            a.imshow(v); a.set_title(title, fontsize=8)
        ax[0].imshow(img); ax[0].set_title(f"{r.image_id[:20]} ({r.group})", fontsize=8)
        show_lab(ax[1], lab[..., 0], lab[..., 1], "membrane: 1 red / 0 grey / ignore amber")
        ax[2].imshow(lab[..., 1], cmap="Greys_r", vmin=0, vmax=255); ax[2].set_title("membrane confidence", fontsize=8)
        show_lab(ax[3], lab[..., 2], lab[..., 3], "evidence (OOF SANC)")
        for a in ax: a.axis("off")
    savefig(fig, "fig09_pseudo_labels")
pseudo_label_figure()

# %% [markdown]
# ## 13. Segmentation student: MSA-U-Net-TR versus U-Net, and loss ablations
# The architecture keeps the original idea, now implemented correctly:
# * a full **decoder** that outputs at full work resolution, with a full-resolution stem so thin membranes are not lost
# * attention-gated skip connections (Oktay 2018)
# * a 2-layer transformer at the stride-32 bottleneck
# * two heads: **membrane** and **evidence region**
#
# Training uses confidence-weighted BCE plus soft Dice over **non-ignored** pixels only. Augmentation is GPU stain augmentation (HED jitter),
# flips and rotations, and mild blur and contrast changes.
#
# Ablations, all at the same epoch budget:
# * a plain U-Net
# * *naive* hard labels, where ignore is treated as 0 and confidence is ignored
# * ignore regions without confidence weights
#
# The held-out fold 0 crops measure **agreement with pseudo-labels**. That agreement is not accuracy.

# %%
class SegCropDS(torch.utils.data.Dataset):
    def __init__(self, df): self.df = df.reset_index(drop=True)
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = cv2.cvtColor(cv2.imread(r.crop), cv2.COLOR_BGR2RGB); lab = cv2.imread(r.label, cv2.IMREAD_UNCHANGED)
        return torch.from_numpy(img), torch.from_numpy(lab)

M_STAIN = torch.tensor(CALIB["M"], dtype=torch.float32, device=DEVICE); M_STAIN_INV = torch.linalg.pinv(M_STAIN)
def hed_jitter(x, sa=0.05, sb=0.02, gen=None):
    """x: (B,3,H,W) float in [0,1]; Tellez-style stain augmentation in the calibrated stain space."""
    B = x.shape[0]
    od = -torch.log(x.clamp(min=1 / 255.0))
    c = torch.einsum("bchw,cd->bdhw", od, M_STAIN_INV)
    a = 1 + (torch.rand(B, 3, 1, 1, device=x.device, generator=gen) * 2 - 1) * sa
    b = (torch.rand(B, 3, 1, 1, device=x.device, generator=gen) * 2 - 1) * sb
    od2 = torch.einsum("bdhw,dc->bchw", (c * a + b).clamp(min=0), M_STAIN)
    return torch.exp(-od2).clamp(0, 1)

def gpu_augment(img, lab):
    if random.random() < 0.5: img, lab = img.flip(-1), lab.flip(-1)
    if random.random() < 0.5: img, lab = img.flip(-2), lab.flip(-2)
    k = random.randint(0, 3); img, lab = torch.rot90(img, k, (-2, -1)), torch.rot90(lab, k, (-2, -1))
    img = hed_jitter(img)
    if random.random() < 0.5:
        img = ((img - 0.5) * (1 + (random.random() - 0.5) * 0.2) + 0.5 + (random.random() - 0.5) * 0.06).clamp(0, 1)
    if random.random() < 0.2:
        g, _, _ = _gauss_1d(random.uniform(0.4, 1.0), img.device)
        img = torch.stack([sep_conv(img[:, c:c + 1], g, g)[:, 0] for c in range(3)], 1)
    return img, lab

def cbr(i, o): return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))
class AttnGate(nn.Module):
    def __init__(self, cg, cx, ci):
        super().__init__(); self.wg = nn.Conv2d(cg, ci, 1); self.wx = nn.Conv2d(cx, ci, 1); self.psi = nn.Conv2d(ci, 1, 1)
    def forward(self, g, x): return x * torch.sigmoid(self.psi(F.relu(self.wg(g) + self.wx(x))))
class BottleneckTR(nn.Module):
    def __init__(self, c, d=256, layers=2):
        super().__init__(); self.inp = nn.Conv2d(c, d, 1); self.pos = nn.Conv2d(d, d, 3, padding=1, groups=d)
        self.tf = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, 8, 2 * d, 0.1, batch_first=True, norm_first=True), layers)
        self.out = nn.Conv2d(d, c, 1)
    def forward(self, x):
        y = self.inp(x); y = y + self.pos(y); B, C, H, W = y.shape
        y = self.tf(y.flatten(2).transpose(1, 2)).transpose(1, 2).reshape(B, C, H, W)
        return x + self.out(y)
class SegNet(nn.Module):
    def __init__(self, enc_name, gates=True, tr=True):
        super().__init__()
        try: self.enc = timm.create_model(enc_name, pretrained=True, features_only=True)
        except Exception:
            print("!! segmentation encoder: pretrained weights unavailable, using random init"); self.enc = timm.create_model(enc_name, pretrained=False, features_only=True)
        ch = self.enc.feature_info.channels(); red = self.enc.feature_info.reduction()
        assert red[0] == 2 and len(ch) == 5, f"expects 5 encoder stages starting at stride 2, got {red}"
        self.tr = BottleneckTR(ch[-1]) if tr else nn.Identity()
        dec = [256, 128, 64, 32]; self.gates = nn.ModuleList(); self.blocks = nn.ModuleList(); prev = ch[-1]
        for i, d in enumerate(dec):
            skip = ch[-2 - i]
            self.gates.append(AttnGate(prev, skip, max(16, skip // 2)) if gates else nn.Identity())
            self.blocks.append(nn.Sequential(cbr(prev + skip, d), cbr(d, d))); prev = d
        self.stem = nn.Sequential(cbr(3, 16), cbr(16, 16))
        self.final = nn.Sequential(cbr(prev + 16, 32), nn.Dropout2d(0.1))
        self.head_mem = nn.Conv2d(32, 1, 1); self.head_ev = nn.Conv2d(32, 1, 1)
        self.register_buffer("mean", torch.tensor(IMNET_MEAN).view(1, 3, 1, 1)); self.register_buffer("std", torch.tensor(IMNET_STD).view(1, 3, 1, 1))
        self.use_gates = gates
    def forward(self, x):
        xn = (x - self.mean) / self.std; f = self.enc(xn); y = self.tr(f[-1])
        for i, blk in enumerate(self.blocks):
            s = f[-2 - i]; y = F.interpolate(y, size=s.shape[-2:], mode="bilinear", align_corners=False)
            s = self.gates[i](y, s) if self.use_gates else s
            y = blk(torch.cat([y, s], 1))
        y = F.interpolate(y, size=x.shape[-2:], mode="bilinear", align_corners=False)
        y = self.final(torch.cat([y, self.stem(x)], 1))
        return self.head_mem(y)[:, 0], self.head_ev(y)[:, 0]

def seg_loss(lm, le, lab, mode):
    ml, mc, el, ec = lab[..., 0], lab[..., 1].float() / 255, lab[..., 2], lab[..., 3].float() / 255
    def weights(l, c):
        valid = (l != 255).float()
        if mode == "naive": return torch.ones_like(valid), (l == 1).float()
        if mode == "ignore": return valid, (l == 1).float()
        return valid * c, (l == 1).float()
    wm, tm = weights(ml, mc); we, te = weights(el, ec)
    bce_m = (wm * F.binary_cross_entropy_with_logits(lm, tm, reduction="none")).sum() / wm.sum().clamp(min=1)
    p = torch.sigmoid(lm); dice = 1 - (2 * (wm * p * tm).sum() + 1) / ((wm * p).sum() + (wm * tm).sum() + 1)
    bce_e = (we * F.binary_cross_entropy_with_logits(le, te, reduction="none")).sum() / we.sum().clamp(min=1)
    return bce_m + dice + CFG.W_EV * bce_e

@torch.no_grad()
def seg_agreement(model, dl):
    model.eval(); tp = fp = fn = 0; tph = fph = fnh = 0; ev_s, ev_y = [], []
    for img, lab in dl:
        img = img.to(DEVICE).permute(0, 3, 1, 2).float() / 255; lab = lab.to(DEVICE)
        with autocast(): lm, le = model(img)
        pm = torch.sigmoid(lm.float()) >= 0.5; ml, mc = lab[..., 0], lab[..., 1]
        v = ml != 255; t = ml == 1; hc = v & (mc >= 191)
        tp += (pm & t & v).sum().item(); fp += (pm & ~t & v).sum().item(); fn += (~pm & t & v).sum().item()
        tph += (pm & t & hc).sum().item(); fph += (pm & ~t & hc).sum().item(); fnh += (~pm & t & hc).sum().item()
        el = lab[..., 2]; ve = el != 255
        if ve.any():
            s = torch.sigmoid(le.float())[ve]; y = (el[ve] == 1)
            k = torch.randperm(len(s), device=s.device)[:20000]; ev_s.append(s[k].cpu().numpy()); ev_y.append(y[k].cpu().numpy())
    d = lambda a, b, c: 2 * a / max(2 * a + b + c, 1)
    ev_s = np.concatenate(ev_s) if ev_s else np.array([]); ev_y = np.concatenate(ev_y) if ev_y else np.array([])
    return dict(agree_Dice=d(tp, fp, fn), agree_IoU=tp / max(tp + fp + fn, 1), agree_precision=tp / max(tp + fp, 1), agree_recall=tp / max(tp + fn, 1),
                agree_Dice_highconf=d(tph, fph, fnh), evidence_AUROC_vs_pseudo=roc_auc_score(ev_y, ev_s) if len(set(ev_y.tolist())) > 1 else np.nan)

cv2.setNumThreads(0)   # OpenCV's own thread pool inside forked DataLoader workers can dead-lock -> disable it

def _seg_loader(df, shuffle, workers):
    kw = dict(batch_size=CFG.SEG_BS, shuffle=shuffle, drop_last=shuffle, num_workers=workers)
    if workers > 0:   # a frozen worker raises after 180 s instead of hanging forever
        kw.update(timeout=180, worker_init_fn=lambda _: cv2.setNumThreads(0), persistent_workers=False)
    return torch.utils.data.DataLoader(SegCropDS(df), **kw)

# Epoch selection (v2.1.2). v2.1 kept the epoch with the best agreement on HIGH-CONFIDENCE pixels only. Those are the easiest
# pixels, so an under-trained epoch 1 could win (it did on Kaggle: epoch 1 of 10). The rule is now: best agreement over ALL
# labelled held-out pixels (rounded to 3 decimals), with high-confidence agreement only as a tie-break. A model trained under the
# old rule is reused only if the new rule picks the same epoch; otherwise it is retrained (deterministically) once.
SEL_RULE = "overall-agreement-v2"
def _sel_key(h): return (round(float(h["agree_Dice"]), 3), float(h["agree_Dice_highconf"]))
def _old_key(h): return float(h["agree_Dice_highconf"])

def train_seg(name, gates, tr, mode, epochs, snapshot_at=None):
    """Checkpoints every epoch (resumes after any interruption); keeps the epoch with the best overall pseudo-label
    agreement on held-out fold-0 crops (tie-break: high-confidence agreement); single-process loading."""
    path = OUT / "models" / f"seg_{slug(name)}.pt"; mpath = CACHE / f"seg_{slug(name)}_metrics.json"
    ckpt_path = CACHE / f"seg_{slug(name)}_ckpt.pt"
    seed_everything(_img_seed(name))                         # same initial weights whether or not earlier stages were cached
    model = SegNet(CFG.SEG_ENCODER, gates, tr).to(DEVICE)
    best_path = OUT / "models" / f"seg_{slug(name)}_best.pt"
    if path.exists() and mpath.exists():
        res = json.load(open(mpath))
        if res.get("selection") != SEL_RULE:                       # trained under the v2.1 rule
            new_ep = max(res["history"], key=_sel_key)["epoch"]
            if new_ep == res["final"]["epoch"]:
                res["selection"] = SEL_RULE; save_json(res, mpath)
                print(f"  {name[:40]}: new selection rule picks the same epoch ({new_ep}); trained model reused")
            else:
                print(f"  {name[:40]}: new selection rule picks epoch {new_ep}, not {res['final']['epoch']}; retraining once")
                res = None
                for f_ in (path, mpath, ckpt_path, best_path): f_.unlink(missing_ok=True)
        if res is not None:
            model.load_state_dict(torch.load(path, map_location=DEVICE)); return model.eval(), res
    tr_df, va_df = CROPS[CROPS.fold != 0], CROPS[CROPS.fold == 0]
    workers = 0   # single-process loading: the v1 run froze twice inside forked loader workers
    dl, dlv = _seg_loader(tr_df, True, workers), _seg_loader(va_df, False, workers)
    opt = torch.optim.AdamW(model.parameters(), lr=CFG.SEG_LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=CFG.SEG_LR, total_steps=max(1, epochs * len(dl)), pct_start=0.1)
    scaler = (torch.amp.GradScaler("cuda") if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler") else torch.cuda.amp.GradScaler()) if DEVICE.type == "cuda" else torch.cuda.amp.GradScaler(enabled=False)
    hist, start, best = [], 0, (-1.0, -1.0)
    if ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        ok = True
        if ck.get("rule") != SEL_RULE and ck["hist"]:               # checkpoint written under the v2.1 rule
            old_ep = ck.get("best_epoch") or max(ck["hist"], key=_old_key)["epoch"]
            ok = max(ck["hist"], key=_sel_key)["epoch"] == old_ep   # saved best weights are valid only if both rules agree
        if ok:
            model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"]); scaler.load_state_dict(ck["scaler"])
            hist, start = ck["hist"], ck["epoch"]; best = _sel_key(max(hist, key=_sel_key)) if hist else (-1.0, -1.0)
            print(f"  resuming {name[:40]} from epoch {start + 1}/{epochs} (best overall agreement so far {best[0]:.3f})")
        else:
            print(f"  {name[:40]}: checkpoint used the old selection rule and the best epoch differs; restarting this model")
            ckpt_path.unlink(missing_ok=True); best_path.unlink(missing_ok=True)
            seed_everything(_img_seed(name)); model = SegNet(CFG.SEG_ENCODER, gates, tr).to(DEVICE)
            opt = torch.optim.AdamW(model.parameters(), lr=CFG.SEG_LR, weight_decay=1e-4)
            sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=CFG.SEG_LR, total_steps=max(1, epochs * len(dl)), pct_start=0.1)
    ep = start
    while ep < epochs:
        seed_everything(_img_seed(name) + 1000 + ep)          # per-epoch seed: shuffling/augmentation identical after a resume
        model.train(); tl = []
        try:
            for img, lab in tqdm(dl, desc=f"{name[:28]} ep{ep + 1}/{epochs}", leave=False):
                img = img.to(DEVICE, non_blocking=True).permute(0, 3, 1, 2).float() / 255; lab = lab.to(DEVICE, non_blocking=True).permute(0, 3, 1, 2)
                img, lab = gpu_augment(img, lab); lab = lab.permute(0, 2, 3, 1)
                with autocast():
                    lm, le = model(img)
                loss = seg_loss(lm.float(), le.float(), lab, mode)
                opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0); scaler.step(opt); scaler.update()
                if sched.last_epoch < sched.total_steps - 1: sched.step()
                tl.append(loss.item())
            m = seg_agreement(model, dlv)
        except RuntimeError as e:
            if workers == 0 or "DataLoader" not in str(e) and "timed out" not in str(e): raise
            print(f"  !! data-loader worker froze ({str(e)[:80]}); switching to single-process loading and repeating epoch {ep + 1}")
            workers = 0; dl, dlv = _seg_loader(tr_df, True, 0), _seg_loader(va_df, False, 0); continue
        m["epoch"] = ep + 1; m["train_loss"] = float(np.mean(tl)); hist.append(m)
        print(f"  {name[:40]} ep{ep + 1}: loss {m['train_loss']:.3f} | pseudo-agreement Dice {m['agree_Dice']:.3f} (high-conf {m['agree_Dice_highconf']:.3f}) | evidence AUROC {m['evidence_AUROC_vs_pseudo']:.3f}", flush=True)
        if _sel_key(m) > best:
            best = _sel_key(m); atomic_torch_save(model.state_dict(), best_path)
        ep += 1
        atomic_torch_save(dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), scaler=scaler.state_dict(),
                        hist=hist, epoch=ep, best=best, best_epoch=max(hist, key=_sel_key)["epoch"], rule=SEL_RULE), ckpt_path)
    model.load_state_dict(torch.load(best_path, map_location=DEVICE))
    best_m = max(hist, key=_sel_key)
    print(f"  {name[:40]}: keeping epoch {best_m['epoch']} of {len(hist)} (overall agreement {best_m['agree_Dice']:.3f}, "
          f"high-conf {best_m['agree_Dice_highconf']:.3f})")
    atomic_torch_save(model.state_dict(), path)
    res = dict(final=best_m, selection=SEL_RULE, last_epoch=hist[-1], history=hist, same_budget=hist[min(len(hist), snapshot_at or len(hist)) - 1])
    save_json(res, mpath); ckpt_path.unlink(missing_ok=True)
    best_path.unlink(missing_ok=True)   # identical to the final file; saves ~100 MB per model
    return model.eval(), res

SEG_EXPS = {
    "MSA-U-Net-TR + ignore + confidence (proposed)": dict(gates=True, tr=True, mode="conf", epochs=CFG.SEG_EPOCHS),
    "U-Net + ignore + confidence": dict(gates=False, tr=False, mode="conf", epochs=CFG.SEG_ABL_EPOCHS),
    "MSA-U-Net-TR + naive hard labels": dict(gates=True, tr=True, mode="naive", epochs=CFG.SEG_ABL_EPOCHS),
    "MSA-U-Net-TR + ignore only": dict(gates=True, tr=True, mode="ignore", epochs=CFG.SEG_ABL_EPOCHS),
}
SEG_PROPOSED = "MSA-U-Net-TR + ignore + confidence (proposed)"
SEG_MODELS, SEG_METRICS = {}, {}
for name, e in SEG_EXPS.items():
    SEG_MODELS[name], SEG_METRICS[name] = train_seg(name, e["gates"], e["tr"], e["mode"], e["epochs"],
                                                    snapshot_at=CFG.SEG_ABL_EPOCHS if name == SEG_PROPOSED else None)
SEGT = pd.DataFrame([{"model": n, **{f"same-budget {k}": v for k, v in m["same_budget"].items() if k.startswith(("agree", "evidence"))},
                      "final_epochs": m["final"]["epoch"], "final agree_Dice": m["final"]["agree_Dice"]} for n, m in SEG_METRICS.items()])
SEGT.to_csv(TAB / "seg_pseudo_agreement.csv", index=False)
print("Held-out agreement with PSEUDO-labels (not accuracy):\n", SEGT.round(3).to_string(index=False))
section_time("segmentation students")
