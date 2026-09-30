"""PARAKH pilot, 30 Sept 2026. Two small, honest measurements for the Stage 1 deck.

A. Inference seal: Ed25519 over RFC 8785 canonical JSON, verifier with nonce store,
   expected sequence and a signed checkpoint. Tampering, replay and tail deletion.
B. Contributor test: k-NN label-disagreement detector on scikit-learn digits
   (public, bundled offline), 10 simulated contributors, exact permutation
   (hypergeometric) test per contributor, Benjamini-Hochberg at q = 0.05.
"""
import hashlib, json, os, random, secrets
import numpy as np
import rfc8785
from nacl.signing import SigningKey
from nacl.exceptions import BadSignatureError
from scipy.stats import hypergeom
from sklearn.datasets import load_digits
from sklearn.neighbors import KNeighborsClassifier

def bh(p, q=0.05):
    p = np.asarray(p); m = len(p); o = np.argsort(p)
    ok = p[o] <= q * np.arange(1, m + 1) / m
    rej = np.zeros(m, bool)
    if ok.any(): rej[o[:np.max(np.where(ok)[0]) + 1]] = True
    return rej

rng = random.Random(2026)
H = lambda b: hashlib.sha256(b).hexdigest()

# ---------------- A. Inference seal ----------------
sk = SigningKey.generate(); vk = sk.verify_key
MODEL = H(b"model-weights-v1")
PRE = {"resize": [224, 224], "norm": "imagenet", "version": "1.0"}

def make_record(seq):
    img = os.urandom(2048)
    body = {"seq": seq, "nonce": secrets.token_hex(16), "image_sha256": H(img),
            "model_sha256": MODEL, "preprocess": PRE,
            "output": {"label": rng.randrange(10), "score": round(rng.random(), 6),
                       "box": [round(rng.random(), 4) for _ in range(4)]}}
    sig = sk.sign(rfc8785.dumps(body)).signature.hex()
    return {"body": body, "sig": sig}, img

class Verifier:
    def __init__(self): self.nonces, self.next_seq = set(), 0
    def check(self, rec, img):
        body = rec["body"]
        try: vk.verify(rfc8785.dumps(body), bytes.fromhex(rec["sig"]))
        except BadSignatureError: return "FAIL:signature"
        if H(img) != body["image_sha256"]: return "FAIL:image"
        if body["model_sha256"] != MODEL: return "FAIL:model"
        if body["nonce"] in self.nonces: return "FAIL:replay"
        if body["seq"] != self.next_seq: return "FAIL:sequence"
        self.nonces.add(body["nonce"]); self.next_seq += 1
        return "PASS"

N = 1000
records = [make_record(i) for i in range(N)]
v = Verifier()
clean_pass = sum(v.check(r, i) == "PASS" for r, i in records)

def tamper(rec, img):
    rec = json.loads(json.dumps(rec)); b = rec["body"]; kind = rng.randrange(6)
    if kind == 0: b["output"]["label"] = (b["output"]["label"] + 1) % 10
    elif kind == 1: b["output"]["score"] = round(min(1, b["output"]["score"] + 0.01), 6)
    elif kind == 2: b["output"]["box"][0] = round(b["output"]["box"][0] + 0.0001, 4)
    elif kind == 3: b["model_sha256"] = H(b"swapped-model")
    elif kind == 4: b["preprocess"] = dict(b["preprocess"], version="1.1")
    else:
        img = bytearray(img); img[rng.randrange(len(img))] ^= 1; img = bytes(img)
    return rec, img

caught_edit = 0
for r, i in records:
    t, ti = tamper(r, i)
    fresh = Verifier(); fresh.next_seq = r["body"]["seq"]
    caught_edit += fresh.check(t, ti) != "PASS"

replays = rng.sample(records, 200)
caught_replay = sum(v.check(r, i) == "FAIL:replay" for r, i in replays)

# Signed checkpoint over the chain head, exported off-machine; then delete a tail.
def chain_head(recs):
    h = "0" * 64
    for r, _ in recs: h = H((h + r["sig"]).encode())
    return h
ckpt = {"count": N, "head": chain_head(records)}
ckpt_sig = sk.sign(rfc8785.dumps(ckpt)).signature
tail_cases, tail_caught = 100, 0
for _ in range(tail_cases):
    k = rng.randint(1, 20); kept = records[:-k]
    vk.verify(rfc8785.dumps(ckpt), ckpt_sig)
    tail_caught += (len(kept) != ckpt["count"]) or (chain_head(kept) != ckpt["head"])

print("A. SEAL")
print(f"  clean records accepted      {clean_pass}/{N}")
print(f"  one-field edits rejected     {caught_edit}/{N}")
print(f"  replays rejected             {caught_replay}/200")
print(f"  tail deletions detected      {tail_caught}/{tail_cases}")

# ---------------- B. Contributor test ----------------
X, y = load_digits(return_X_y=True)
N_CONTRIB, N_PER = 10, 89

def run(flip_rate, seed, poisoned=True):
    r = np.random.default_rng(seed)
    idx = r.permutation(len(y))
    ref, held, pool = idx[:600], idx[600:900], idx[900:900 + N_CONTRIB * N_PER]
    knn = KNeighborsClassifier(n_neighbors=7).fit(X[ref], y[ref])
    # threshold: flag when the k-NN vote for the given label is <= t; t set on ref (leave-one-out)
    loo = KNeighborsClassifier(n_neighbors=8).fit(X[ref], y[ref])
    nb = loo.kneighbors(X[ref], return_distance=False)[:, 1:]
    ref_vote = (y[ref][nb] == y[ref][:, None]).mean(1)
    t = np.quantile(ref_vote, 0.05)                       # calibration split
    def flags(Xs, ys):
        nbi = knn.kneighbors(Xs, return_distance=False)
        return (y[ref][nbi] == ys[:, None]).mean(1) <= t
    fpr_heldout = flags(X[held], y[held]).mean()           # held-out split
    Xc, yc = X[pool].copy(), y[pool].copy()
    owner = np.repeat(np.arange(N_CONTRIB), N_PER)
    truth = np.zeros(len(yc), bool)
    if poisoned:                                           # contributor 2 flips labels
        m = np.where(owner == 2)[0]; k = int(round(flip_rate * len(m)))
        f = r.choice(m, k, replace=False)
        yc[f] = (yc[f] + r.integers(1, 10, k)) % 10; truth[f] = True
    b = np.where(owner == 5)[0]                            # contributor 5: honest, dimmer camera
    Xc[b] = np.clip(Xc[b] * 0.85, 0, 16)
    fl = flags(Xc, yc)
    total, K = len(fl), fl.sum()
    p = [hypergeom.sf(fl[owner == c].sum() - 1, total, K, N_PER) for c in range(N_CONTRIB)]
    rej = bh(p)
    return fpr_heldout, rej, fl, truth, owner

print("\nB. CONTRIBUTOR TEST (sklearn digits, 10 contributors x 89 images, BH q=0.05)")
S = 200
fprs = []
for rate in [0.02, 0.05, 0.10, 0.15, 0.20]:
    det = clean_flag = 0; recalls = []
    for s in range(S):
        f, rej, fl, tr, ow = run(rate, s); fprs.append(f)
        det += rej[2]; clean_flag += rej[[c for c in range(10) if c != 2]].any()
        if tr.any(): recalls.append(fl[tr].mean())
    print(f"  flip {rate:4.0%}: poisoned contributor flagged {det/S:6.1%} | any clean contributor flagged {clean_flag/S:5.1%} | sample recall {np.mean(recalls):5.1%}")
allclean = sum(run(0, s, poisoned=False)[1].any() for s in range(S))
honest5 = sum(run(0, s, poisoned=False)[1][5] for s in range(S))
print(f"  all-clean runs with any contributor flagged: {allclean}/{S} ({allclean/S:.1%}); dimmer-camera contributor flagged {honest5}/{S}")
print(f"  per-detector false-positive rate on held-out clean split: {np.mean(fprs):.1%}")

# ---------------- C. Edit-and-rehash without the key, + Wilson intervals ----------------
def edit_and_rehash(rec, img):
    """Attacker changes the image AND recomputes every hash in the record, but cannot re-sign."""
    rec = json.loads(json.dumps(rec)); b = rec["body"]
    img = bytearray(img); img[rng.randrange(len(img))] ^= 0xFF; img = bytes(img)
    b["image_sha256"] = H(img)
    b["output"]["label"] = (b["output"]["label"] + 3) % 10
    rec["sig"] = SigningKey.generate().sign(rfc8785.dumps(b)).signature.hex()  # signed with a wrong key
    return rec, img
caught_rehash = 0
for r, i in records:
    t, ti = edit_and_rehash(r, i)
    fresh = Verifier(); fresh.next_seq = r["body"]["seq"]
    caught_rehash += fresh.check(t, ti) != "PASS"
print(f"\nC. edit-and-rehash with attacker's own key rejected: {caught_rehash}/{N}")
total = caught_edit + caught_rehash + caught_replay + tail_caught
print(f"   all tampering trials rejected: {total}/{N + N + 200 + tail_cases}")

def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h
for lab, k in (("all-clean false alarm 6/200", 6), ("15% poison caught 191/200", 191)):
    lo, hi = wilson(k, 200); print(f"   {lab}: 95% Wilson CI {lo:.1%} to {hi:.1%}")
