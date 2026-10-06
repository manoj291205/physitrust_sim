"""
PhysiTrust-FL simulation
Grid-challenged proof-of-presence (PoP), hardware-bound measurement provenance
and DID-bound trust management for federated anomaly detection on the
IEEE 33-bus distribution feeder.

Author : Manojkumar Ashokkumar (VIT Vellore)
Needs  : Python 3.9+, numpy, scipy, matplotlib, eth-account, cryptography
Install: pip install numpy scipy matplotlib eth-account cryptography
Run    : python physitrust_sim.py            (full run, about 30 min on one CPU core)
         python physitrust_sim.py --quick    (smaller run, about 3 min)
Output : figures (PNG) and tables (CSV) in ./results
"""
import argparse, hashlib, json, os, time
import numpy as np
from scipy.stats import chi2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from eth_account import Account
from eth_account.messages import encode_typed_data
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec

OUT = "results"
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.size": 8,
                     "axes.titlesize": 8, "legend.fontsize": 6.5})

# ---------------------------------------------------------------------------
# 1. IEEE 33-bus feeder (Baran and Wu) and vectorised backward/forward sweep
# ---------------------------------------------------------------------------
BRANCH = np.array([  # from, to, R (ohm), X (ohm)
    [1, 2, .0922, .0470], [2, 3, .4930, .2511], [3, 4, .3660, .1864],
    [4, 5, .3811, .1941], [5, 6, .8190, .7070], [6, 7, .1872, .6188],
    [7, 8, .7114, .2351], [8, 9, 1.030, .7400], [9, 10, 1.044, .7400],
    [10, 11, .1966, .0650], [11, 12, .3744, .1238], [12, 13, 1.468, 1.155],
    [13, 14, .5416, .7129], [14, 15, .5910, .5260], [15, 16, .7463, .5450],
    [16, 17, 1.289, 1.721], [17, 18, .7320, .5740], [2, 19, .1640, .1565],
    [19, 20, 1.5042, 1.3554], [20, 21, .4095, .4784], [21, 22, .7089, .9373],
    [3, 23, .4512, .3083], [23, 24, .8980, .7091], [24, 25, .8960, .7011],
    [6, 26, .2030, .1034], [26, 27, .2842, .1447], [27, 28, 1.059, .9337],
    [28, 29, .8042, .7006], [29, 30, .5075, .2585], [30, 31, .9744, .9630],
    [31, 32, .3105, .3619], [32, 33, .3410, .5302]])
LOAD_KW = np.array([100, 90, 120, 60, 60, 200, 200, 60, 60, 45, 60, 60, 120,
                    60, 60, 60, 90, 90, 90, 90, 90, 90, 420, 420, 60, 60, 60,
                    120, 200, 150, 210, 60], float)
LOAD_KVAR = np.array([60, 40, 80, 30, 20, 100, 100, 20, 20, 30, 35, 35, 80,
                      10, 20, 20, 40, 40, 40, 40, 40, 50, 200, 200, 25, 25,
                      20, 70, 600, 70, 100, 40], float)
VBASE_KV, SBASE_MVA = 12.66, 1.0
ZBASE = VBASE_KV ** 2 / SBASE_MVA
NB, NBR = 33, 32
FB = BRANCH[:, 0].astype(int) - 1
TB = BRANCH[:, 1].astype(int) - 1
ZBR = (BRANCH[:, 2] + 1j * BRANCH[:, 3]) / ZBASE
P_NOM = np.r_[0.0, LOAD_KW / 1000.0]       # per unit on 1 MVA
Q_NOM = np.r_[0.0, LOAD_KVAR / 1000.0]
ACTUATORS = np.array([18, 22, 25, 33]) - 1  # smart-inverter buses (feeder ends)
NODES = np.arange(1, NB)                     # one edge node per load bus 2..33
# branch k ends at bus TB[k] = k+1, so the upstream branch of bus b is b-1


def power_flow(P, Q, iters=30, tol=1e-10):
    """Backward/forward sweep for arrays P, Q of shape (T, 33) in p.u.
    Returns complex bus voltages (T, 33) and receiving-end branch flows."""
    T = P.shape[0]
    V = np.ones((T, NB), complex)
    for _ in range(iters):
        Ibus = np.conj((P + 1j * Q) / V)
        Ibr = np.zeros((T, NBR), complex)
        for k in range(NBR - 1, -1, -1):          # backward sweep
            Ibr[:, k] = Ibus[:, TB[k]]
            Ibus[:, FB[k]] += Ibr[:, k]
        Vn = np.ones((T, NB), complex)
        for k in range(NBR):                      # forward sweep
            Vn[:, TB[k]] = Vn[:, FB[k]] - ZBR[k] * Ibr[:, k]
        done = np.max(np.abs(Vn - V)) < tol
        V = Vn
        if done:
            break
    Sflow = V[:, TB] * np.conj(Ibr)
    return V, Sflow


def electrical_distance():
    """|Z| of the path between every pair of buses (ohm)."""
    parent = {int(TB[k]): int(FB[k]) for k in range(NBR)}
    zb = {int(TB[k]): abs(BRANCH[k, 2] + 1j * BRANCH[k, 3]) for k in range(NBR)}
    def to_root(b):
        p = [b]
        while b in parent:
            b = parent[b]
            p.append(b)
        return p
    paths = [to_root(b) for b in range(NB)]
    D = np.zeros((NB, NB))
    for i in range(NB):
        for j in range(NB):
            lca = next(b for b in paths[i] if b in paths[j])
            up_i = paths[i][:paths[i].index(lca)]
            up_j = paths[j][:paths[j].index(lca)]
            D[i, j] = sum(zb[b] for b in up_i) + sum(zb[b] for b in up_j)
    return D


# ---------------------------------------------------------------------------
# 2. Grid-challenged proof-of-presence (PoP)
# ---------------------------------------------------------------------------
def keccak_hex(b: bytes) -> str:
    from eth_utils import keccak
    return keccak(b).hex()


def chips_from_seed(seed: bytes, n_act: int, L: int) -> np.ndarray:
    """Pseudo-random +/-1 spreading codes, one row per actuator."""
    C = np.zeros((n_act, L))
    for a in range(n_act):
        bits, ctr = [], 0
        while len(bits) < L:
            h = hashlib.sha256(seed + bytes([a]) + ctr.to_bytes(4, "big")).digest()
            for byte in h:
                for s in range(8):
                    bits.append((byte >> s) & 1)
            ctr += 1
        C[a] = 2 * np.array(bits[:L]) - 1
    return C


class Feeder:
    """Generates 1 Hz measurement windows with load noise and optional dither."""
    def __init__(self, rng, sigma_load=0.02, phi=0.95, sigma_v=5e-4,
                 sigma_q_rel=2e-3, sigma_q_abs=2e-4):
        self.rng, self.sl, self.phi = rng, sigma_load, phi
        self.sv, self.sqr, self.sqa = sigma_v, sigma_q_rel, sigma_q_abs

    def window(self, L, codes=None, dq=0.03, m0=None, acts=ACTUATORS):
        rng = self.rng
        m0 = rng.uniform(0.4, 1.0) if m0 is None else m0
        e = np.zeros((L, NB, 2))
        e[0] = rng.normal(0, self.sl, (NB, 2))
        for t in range(1, L):  # AR(1) load fluctuation, separate for P and Q
            e[t] = self.phi * e[t - 1] + np.sqrt(1 - self.phi ** 2) * \
                rng.normal(0, self.sl, (NB, 2))
        P = m0 * P_NOM[None] * (1 + e[..., 0])
        Q = m0 * Q_NOM[None] * (1 + e[..., 1])
        if codes is not None:  # inverter injects dq*c(t) of reactive power
            Q[:, acts] -= dq * codes.T
        V, S = power_flow(P, Q)
        Vm = np.abs(V) + rng.normal(0, self.sv, (L, NB))
        Qf = S.imag
        Qf = Qf + rng.normal(0, 1, Qf.shape) * (self.sqr * np.abs(Qf) + self.sqa)
        head_p = P.sum(1).mean()
        return Vm, Qf, head_p, m0


def expected_gains(m0, dq, acts=ACTUATORS):
    """Verifier-side fingerprint: response of |V_b| and upstream Q-flow of
    every bus to +1 chip of each actuator, from the utility feeder model."""
    n = len(acts)
    P = np.repeat((m0 * P_NOM)[None], 1 + 2 * n, 0)
    Q = np.repeat((m0 * Q_NOM)[None], 1 + 2 * n, 0)
    for a, bus in enumerate(acts):
        Q[1 + 2 * a, bus] -= dq
        Q[2 + 2 * a, bus] += dq
    V, S = power_flow(P, Q)
    Vm, Qf = np.abs(V), S.imag
    gV = np.zeros((NB, n)); gQ = np.zeros((NBR, n))
    for a in range(n):
        gV[:, a] = (Vm[1 + 2 * a] - Vm[2 + 2 * a]) / 2
        gQ[:, a] = (Qf[1 + 2 * a] - Qf[2 + 2 * a]) / 2
    return gV, gQ


def estimate_gains(y, codes):
    """OLS of a measured series on the codes plus a quadratic trend."""
    L = y.shape[0]
    t = np.linspace(-1, 1, L)
    X = np.column_stack([codes.T, np.ones(L), t, t ** 2])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ beta
    s2 = (res ** 2).sum(0) / (L - X.shape[1])
    XtXi = np.linalg.inv(X.T @ X)
    k = codes.shape[0]
    se = np.sqrt(np.outer(s2, np.diag(XtXi)[:k]).T)  # (k, n_series)
    return beta[:k], se


def pop_statistic(Vm_b, Qf_b, codes, gV_b, gQ_b, kappa=0.02):
    """Chi-square distance between measured and expected fingerprint (E5)."""
    gv_hat, sv = estimate_gains(Vm_b[:, None], codes)
    gq_hat, sq = estimate_gains(Qf_b[:, None], codes)
    g_hat = np.r_[gv_hat[:, 0], gq_hat[:, 0]]
    se = np.r_[sv[:, 0], sq[:, 0]]
    g_exp = np.r_[gV_b, gQ_b]
    var = se ** 2 + (kappa * np.abs(g_exp)) ** 2
    return float(((g_hat - g_exp) ** 2 / var).sum())


def pop_threshold(n_act, alpha=1e-3):
    return chi2.ppf(1 - alpha, 2 * n_act)


POP_THRESHOLD = pop_threshold(len(ACTUATORS))


def m0_from_head(head_p):
    return head_p / P_NOM.sum()


def pop_trial(feeder, L, dq, acts=ACTUATORS, rng=None):
    """One challenge window. Returns measured series, codes and expected
    fingerprints, plus an off-grid window and a stale (replayed) window."""
    rng = rng or np.random.default_rng()
    codes = chips_from_seed(rng.bytes(32), len(acts), L)
    Vm, Qf, head_p, _ = feeder.window(L, codes, dq, acts=acts)
    old = chips_from_seed(rng.bytes(32), len(acts), L)
    Vr, Qr, _, _ = feeder.window(L, old, dq, acts=acts)
    Vo, Qo, _, _ = feeder.window(L, None, dq, acts=acts)
    gV, gQ = expected_gains(m0_from_head(head_p), dq, acts)
    return dict(codes=codes, Vm=Vm, Qf=Qf, Vr=Vr, Qr=Qr, Vo=Vo, Qo=Qo, gV=gV, gQ=gQ)


# ---------------------------------------------------------------------------
# 3. Web3 PKI layer: DID keys, secure-element keys, Merkle provenance and a
#    Python mirror of the on-chain registry / round contract
# ---------------------------------------------------------------------------
ACTIVE, QUARANTINED, REVOKED = "ACTIVE", "QUARANTINED", "REVOKED"
CHAIN_ID = 31337
CONTRACT = "0x5FbDB2315678afecb367f032d93F642f64180aa3"


def sha256(b: bytes) -> bytes:
    return hashlib.sha256(b).digest()


def merkle_root(leaves):
    level = [sha256(x) for x in leaves] or [sha256(b"")]
    while len(level) > 1:
        if len(level) % 2:
            level.append(level[-1])
        level = [sha256(level[i] + level[i + 1]) for i in range(0, len(level), 2)]
    return level[0]


class SecureElement:
    """Emulates a metering secure element (P-256, e.g. ATECC608-class).
    It signs the digest of every raw measurement window it produces."""
    def __init__(self):
        self._sk = ec.generate_private_key(ec.SECP256R1())
        self.pub = self._sk.public_key()
        self.counter = 0

    def sign_window(self, raw: np.ndarray):
        self.counter += 1
        digest = sha256(raw.astype(np.float32).tobytes() +
                        self.counter.to_bytes(8, "big"))
        return digest, self._sk.sign(digest, ec.ECDSA(hashes.SHA256()))


def se_verify(pub, digest, sig):
    try:
        pub.verify(sig, digest, ec.ECDSA(hashes.SHA256()))
        return True
    except Exception:
        return False


def typed_update(addr, rnd, ghash, uhash, mroot, nonce):
    return {
        "types": {
            "EIP712Domain": [{"name": "name", "type": "string"},
                             {"name": "version", "type": "string"},
                             {"name": "chainId", "type": "uint256"},
                             {"name": "verifyingContract", "type": "address"}],
            "Update": [{"name": "did", "type": "address"},
                       {"name": "round", "type": "uint256"},
                       {"name": "globalHash", "type": "bytes32"},
                       {"name": "updateHash", "type": "bytes32"},
                       {"name": "measRoot", "type": "bytes32"},
                       {"name": "nonce", "type": "uint256"}]},
        "primaryType": "Update",
        "domain": {"name": "PhysiTrustFL", "version": "1",
                   "chainId": CHAIN_ID, "verifyingContract": CONTRACT},
        "message": {"did": addr, "round": rnd, "globalHash": ghash,
                    "updateHash": uhash, "measRoot": mroot, "nonce": nonce}}


class EdgeIdentity:
    """DID key (secp256k1, Ethereum account) plus secure-element key."""
    def __init__(self, bus):
        self.acct = Account.create()
        self.addr = self.acct.address
        self.se = SecureElement()
        self.bus = bus
        self.nonce = 0

    def sign_update(self, rnd, ghash, uhash, mroot):
        self.nonce += 1
        msg = encode_typed_data(full_message=typed_update(
            self.addr, rnd, ghash, uhash, mroot, self.nonce))
        return self.nonce, self.acct.sign_message(msg).signature


class Ledger:
    """Python mirror of the PhysiTrust contract logic (registry, rounds,
    commit-reveal challenge seed, update admission, key state, anchoring)."""
    def __init__(self):
        self.reg, self.state, self.used = {}, {}, set()
        self.round, self.seed_commit, self.seed = 0, None, None
        self.global_hash, self.meas_commit, self.log = None, {}, []

    def register(self, ident: EdgeIdentity, vc_ok=True):
        if vc_ok:  # utility-issued credential binds DID, SE key and bus
            self.reg[ident.addr] = (ident.se.pub, ident.bus)
            self.state[ident.addr] = ACTIVE

    def open_round(self, seed_commit, global_hash):
        self.round += 1
        self.seed_commit, self.seed = seed_commit, None
        self.global_hash, self.meas_commit = global_hash, {}

    def commit_measurement(self, addr, digest):
        if self.seed is None:  # commits are only accepted before the reveal
            self.meas_commit[addr] = digest
            return True
        return False

    def reveal_seed(self, seed):
        if keccak_hex(seed) == self.seed_commit:
            self.seed = seed
            return True
        return False

    def admit(self, addr, nonce, sig, uhash, mroot):
        """Signature, key state, freshness and challenge-commit checks (E7)."""
        if addr not in self.reg or self.state[addr] == REVOKED:
            return False, "unregistered or revoked"
        if (addr, nonce) in self.used:
            return False, "replayed nonce"
        if addr not in self.meas_commit:
            return False, "no challenge commitment"
        msg = encode_typed_data(full_message=typed_update(
            addr, self.round, self.global_hash, uhash, mroot, nonce))
        if Account.recover_message(msg, signature=sig) != addr:
            return False, "bad signature"
        self.used.add((addr, nonce))
        return True, "ok"

    def set_state(self, addr, s):
        if self.state.get(addr) != s:
            self.log.append((self.round, addr, s))
            self.state[addr] = s


def benchmark_crypto(n=200):
    """Timing of the cryptographic operations used per node per round."""
    ident = EdgeIdentity(5)
    raw = np.random.default_rng(0).normal(size=(120, 2)).astype(np.float32)
    h32 = "0x" + "11" * 32
    res = {}
    t = time.perf_counter()
    for _ in range(n):
        d, s = ident.se.sign_window(raw)
    res["SE P-256 sign (window digest)"] = (time.perf_counter() - t) / n
    t = time.perf_counter()
    for _ in range(n):
        se_verify(ident.se.pub, d, s)
    res["SE P-256 verify"] = (time.perf_counter() - t) / n
    t = time.perf_counter()
    for _ in range(n):
        nonce, sig = ident.sign_update(1, h32, h32, h32)
    res["EIP-712 secp256k1 sign"] = (time.perf_counter() - t) / n
    msg = encode_typed_data(full_message=typed_update(ident.addr, 1, h32, h32, h32, nonce))
    t = time.perf_counter()
    for _ in range(n):
        Account.recover_message(msg, signature=sig)
    res["EIP-712 ecrecover (verify)"] = (time.perf_counter() - t) / n
    leaves = [os.urandom(32) for _ in range(512)]
    t = time.perf_counter()
    for _ in range(20):
        merkle_root(leaves)
    res["Merkle root of 512 windows"] = (time.perf_counter() - t) / 20
    return {k: v * 1e3 for k, v in res.items()}  # milliseconds


def gas_estimate():
    """Gas per contract call estimated from the EVM fee schedule
    (21000 base, 16 gas per calldata byte, 22100 new SSTORE,
    5000 SSTORE update, 3000 ecrecover, about 1500 per event).
    These are estimates, not measurements on a live chain."""
    base, cd, new, upd, rec, ev = 21000, 16, 22100, 5000, 3000, 1500
    calls = {
        "registerNode (DID, SE key hash, bus, VC hash)": base + 4 * 32 * cd + 3 * new + ev + rec,
        "openRound (seed commit, global hash)": base + 2 * 32 * cd + 2 * upd + ev,
        "commitMeasurement (window digest)": base + 32 * cd + new + ev,
        "revealSeed (32 B)": base + 32 * cd + upd + 2100 + ev,
        "submitUpdate (EIP-712 payload, sig)": base + (6 * 32 + 65) * cd + rec + new + new + ev,
        "setKeyState (quarantine / revoke)": base + 64 * cd + upd + ev,
        "anchorModel (CID hash)": base + 64 * cd + upd + ev,
    }
    return calls


# ---------------------------------------------------------------------------
# 4. Grid-edge telemetry, anomaly injection and the local autoencoder
# ---------------------------------------------------------------------------
W = 10           # window length (minutes)
FEAT = 4         # |V|, P, Q, upstream Q-flow


def daily_shape(h):
    return (0.45 + 0.25 * np.exp(-((h - 8.5) / 2.0) ** 2)
            + 0.45 * np.exp(-((h - 19.5) / 2.5) ** 2))


def make_telemetry(rng, days=6):
    """1-minute time series for all buses from repeated power flows."""
    T = days * 1440
    h = (np.arange(T) % 1440) / 60.0
    shift = rng.uniform(-1.0, 1.0, NB)
    mP = np.stack([daily_shape((h - s) % 24) for s in shift], 1)
    e = np.zeros((T, NB, 2))
    for t in range(1, T):
        e[t] = 0.9 * e[t - 1] + rng.normal(0, 0.02, (NB, 2))
    P = P_NOM[None] * mP * (1 + e[..., 0])
    Q = Q_NOM[None] * mP * (1 + e[..., 1])
    V, S = power_flow(P, Q)
    Vm = np.abs(V) + rng.normal(0, 5e-4, V.shape)
    Qf = S.imag + rng.normal(0, 1e-3, S.shape)
    data = {}
    for b in NODES:
        data[b] = np.stack([Vm[:, b], P[:, b], Q[:, b], Qf[:, b - 1]], 1)
    return data


def windows(x, stride=5):
    idx = np.arange(0, x.shape[0] - W + 1, stride)
    return np.stack([x[i:i + W] for i in idx])        # (n, W, FEAT)


def inject(win, rng):
    """Cyber anomalies on normalised windows (FDI, ramp, scaling, freeze,
    noise burst, physically inconsistent P-V move)."""
    x = win.copy()
    kind = rng.integers(6)
    k = rng.uniform(2.5, 5.0)
    if kind == 0:
        x[:, rng.integers(FEAT)] += k
    elif kind == 1:
        x[:, 1] += np.linspace(0, k, W)
    elif kind == 2:
        x[:, 2] = x[:, 2] * rng.uniform(1.5, 2.5) + 1.0
    elif kind == 3:
        x[:] = x[0] + rng.normal(0, 0.02, x.shape)
        x[:, rng.integers(FEAT)] += 1.5
    elif kind == 4:
        x += rng.normal(0, 1.5, x.shape)
    else:
        x[:, 1] += k
        x[:, 0] += k
    return x


class NodeData:
    def __init__(self, raw, rng):
        n = raw.shape[0]
        tr, va = raw[: 4 * n // 6], raw[4 * n // 6: 5 * n // 6]
        te = raw[5 * n // 6:]
        self.mu, self.sd = tr.mean(0), tr.std(0) + 1e-9
        f = lambda z: (z - self.mu) / self.sd
        self.train = windows(f(tr)).reshape(-1, W * FEAT)
        self.val = windows(f(va)).reshape(-1, W * FEAT)
        tw = windows(f(te))
        lab = rng.random(len(tw)) < 0.5
        for i in np.where(lab)[0]:
            tw[i] = inject(tw[i], rng)
        self.test, self.label = tw.reshape(-1, W * FEAT), lab.astype(int)
        self.fake = np.stack([inject(w, rng) for w in
                              windows(f(tr))[: len(self.train)]]).reshape(-1, W * FEAT)


D_IN, H = W * FEAT, 12
SHAPES = [(H, D_IN), (H,), (D_IN, H), (D_IN,)]
NPARAM = sum(int(np.prod(s)) for s in SHAPES)


def unpack(w):
    out, i = [], 0
    for s in SHAPES:
        n = int(np.prod(s)); out.append(w[i:i + n].reshape(s)); i += n
    return out


def ae_scores(w, X):
    W1, b1, W2, b2 = unpack(w)
    Z = np.tanh(X @ W1.T + b1)
    return ((Z @ W2.T + b2 - X) ** 2).mean(1)


def local_train(w, X, rng, epochs=1, lr=0.05, bs=32):
    w = w.copy()
    for _ in range(epochs):
        perm = rng.permutation(len(X))
        for s in range(0, len(X), bs):
            xb = X[perm[s:s + bs]]
            W1, b1, W2, b2 = unpack(w)
            A = xb @ W1.T + b1
            Z = np.tanh(A)
            R = Z @ W2.T + b2 - xb
            n = len(xb)
            gR = 2 * R / (n * D_IN)
            gW2, gb2 = gR.T @ Z, gR.sum(0)
            gA = (gR @ W2) * (1 - Z ** 2)
            gW1, gb1 = gA.T @ xb, gA.sum(0)
            g = np.concatenate([gW1.ravel(), gb1, gW2.ravel(), gb2])
            w -= lr * g
    return w


def auc(scores, labels):
    order = np.argsort(scores)
    ranks = np.empty(len(scores)); ranks[order] = np.arange(1, len(scores) + 1)
    pos = labels == 1
    n1, n0 = pos.sum(), (~pos).sum()
    return (ranks[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def evaluate(w, nodes):
    s = np.concatenate([ae_scores(w, nd.test) for nd in nodes])
    y = np.concatenate([nd.label for nd in nodes])
    thr = np.percentile(np.concatenate([ae_scores(w, nd.val) for nd in nodes]), 99)
    pred = s > thr
    tp = (pred & (y == 1)).sum(); fp = (pred & (y == 0)).sum()
    fn = (~pred & (y == 1)).sum()
    f1 = 2 * tp / max(2 * tp + fp + fn, 1)
    return auc(s, y), f1


# ---------------------------------------------------------------------------
# 5. Aggregation rules (baselines) and the PhysiTrust-FL round protocol
# ---------------------------------------------------------------------------
def agg_fedavg(U, **k):
    return U.mean(0)


def agg_median(U, **k):
    return np.median(U, 0)


def agg_trimmed(U, beta=0.2, **k):
    n = len(U); t = int(beta * n)
    S = np.sort(U, 0)
    return S[t:n - t].mean(0) if n - 2 * t > 0 else np.median(U, 0)


def agg_multikrum(U, f=0, **k):
    n = len(U)
    f = int(min(f, max((n - 3) // 2, 0)))
    d = ((U[:, None, :] - U[None, :, :]) ** 2).sum(-1)
    score = np.sort(d, 1)[:, 1:n - f - 1].sum(1)
    keep = np.argsort(score)[: max(n - f, 1)]
    return U[keep].mean(0)


def agg_fltrust(U, root=None, **k):
    nr = np.linalg.norm(root) + 1e-12
    cos = U @ root / (np.linalg.norm(U, axis=1) * nr + 1e-12)
    ts = np.maximum(cos, 0)
    if ts.sum() == 0:
        return np.zeros_like(root)
    Un = U * (nr / (np.linalg.norm(U, axis=1, keepdims=True) + 1e-12))
    return (ts[:, None] * Un).sum(0) / ts.sum()


def weighted_median(U, wts):
    """Coordinate-wise weighted median of the rows of U (E9)."""
    idx = np.argsort(U, 0)
    Us = np.take_along_axis(U, idx, 0)
    cw = np.cumsum(wts[idx], 0)
    k = (cw >= 0.5 * wts.sum()).argmax(0)
    return Us[k, np.arange(U.shape[1])]


BASELINES = {"FedAvg": agg_fedavg, "Multi-Krum": agg_multikrum,
             "Median": agg_median, "Trimmed mean": agg_trimmed,
             "FLTrust": agg_fltrust}


class World:
    """Telemetry, node datasets, identities and SE-signed training windows
    for one random seed (built once, reused by every method)."""
    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)
        tel = make_telemetry(self.rng)
        self.nodes = [NodeData(tel[b], self.rng) for b in NODES]
        self.ids = [EdgeIdentity(b) for b in NODES]
        self.signed = []
        for nd, idt in zip(self.nodes, self.ids):
            self.signed.append([idt.se.sign_window(x) for x in nd.train])
        pick = self.rng.choice(len(self.nodes), 3, replace=False)
        self.root = np.concatenate([self.nodes[i].train[:40] for i in pick])
        self.w0 = self.rng.normal(0, 0.1, NPARAM)


def malicious_update(attack, w, nd, rng, gamma=4.0):
    if attack == "DP":       # trains on fabricated anomalies labelled normal
        return local_train(w, nd.fake, rng, epochs=3) - w
    if attack == "ADP":      # adaptive: keeps its loss on real data low so
        n = len(nd.train) // 2   # that a spot-check on genuine windows passes
        X = np.concatenate([nd.fake[:n], nd.train[:n]])
        u = local_train(w, X, rng, epochs=2) - w
        l0 = ae_scores(w, nd.train[:200]).mean()
        for _ in range(10):
            if ae_scores(w + u, nd.train[:200]).mean() <= 1.15 * l0:
                break
            u = 0.5 * u
        return u
    honest = local_train(w, nd.train, rng) - w
    if attack == "SF":       # sign flip with scaling
        return -gamma * honest
    if attack == "GN":       # Gaussian noise of large norm
        return rng.normal(0, 1, NPARAM) * (5 * np.linalg.norm(honest) / np.sqrt(NPARAM))
    return honest


def run_fl(world, method, attack="SF", frac=0.0, n_sybil=0, rounds=30,
           sybil_attack="ADP",
           seed=0, ablate=(), L=120, dq=0.03, c_spot=4, lam=0.6,
           tau_q=0.3, track=False):
    rng = np.random.default_rng(1000 + seed)
    nodes, N = world.nodes, len(world.nodes)
    bad = set(rng.choice(N, int(round(frac * N)), replace=False).tolist())
    feeder = Feeder(np.random.default_rng(2000 + seed))
    w = world.w0.copy()
    proposed = method == "PhysiTrust-FL"
    if proposed:
        ledger = Ledger()
        syb_ids = [EdgeIdentity(int(rng.integers(1, NB))) for _ in range(n_sybil)]
        for idt in world.ids + syb_ids:
            ledger.register(idt)          # Sybil devices hold valid credentials
        T = {i: 0.6 for i in range(N + n_sybil)}
        streak_q = {i: 0 for i in T}; streak_ok = {i: 0 for i in T}
    hist, stats = [], {"pop_reject": 0, "prov_reject": 0, "sig_reject": 0}
    for r in range(rounds):
        # ---- local updates --------------------------------------------------
        U = []
        for i, nd in enumerate(nodes):
            U.append(malicious_update(attack, w, nd, rng) if i in bad
                     else local_train(w, nd.train, rng) - w)
        for j in range(n_sybil):  # Sybils hold copied data and attack
            nd = nodes[int(rng.integers(N))]
            U.append(malicious_update(sybil_attack, w, nd, rng))
        U = np.array(U)
        if not proposed:
            f = len(bad) + n_sybil
            root = local_train(w, world.root, rng) - w if method == "FLTrust" else None
            w = w + BASELINES[method](U, f=f, root=root)
            if track:
                hist.append(evaluate(w, nodes)[0])
            continue
        # ---- PhysiTrust-FL: challenge, commit, reveal, admit, verify ---------
        seed_b = rng.bytes(32)
        ghash = "0x" + sha256(w.tobytes()).hex()
        ledger.open_round(keccak_hex(seed_b), ghash)
        codes = chips_from_seed(seed_b, len(ACTUATORS), L)
        Vm, Qf, head_p, _ = feeder.window(L, codes, dq)          # real feeder
        Vo, Qo, _, _ = feeder.window(L, None, dq)                 # off-grid site
        all_ids = world.ids + syb_ids
        for idt in all_ids:   # measurement digests committed before reveal
            ledger.commit_measurement(idt.addr, b"digest")
        ledger.reveal_seed(seed_b)
        gV, gQ = expected_gains(m0_from_head(head_p), dq)
        checks = []                       # pass 1: PKI, PoP and provenance
        for i, idt in enumerate(all_ids):
            if ledger.state[idt.addr] == REVOKED:
                continue
            uhash = "0x" + sha256(U[i].tobytes()).hex()
            mroot = "0x" + merkle_root([d for d, _ in world.signed[i % N][:64]]).hex()
            nonce, sig = idt.sign_update(ledger.round, ghash, uhash, mroot)
            ok, _ = ledger.admit(idt.addr, nonce, sig, uhash, mroot)
            if not ok:
                stats["sig_reject"] += 1
                continue
            b = idt.bus
            if i < N:
                st = pop_statistic(Vm[:, b], Qf[:, b - 1], codes, gV[b], gQ[b - 1])
            else:          # Sybil device is not on this feeder
                st = pop_statistic(Vo[:, b], Qo[:, b - 1], codes, gV[b], gQ[b - 1])
            pop_ok = ("pop" in ablate) or st < POP_THRESHOLD
            prov_ok = True
            if "prov" not in ablate:   # spot-check SE-signed training windows
                k = i % N
                sel = rng.choice(len(nodes[k].train), c_spot, replace=False)
                pub = idt.se.pub if i < N else None
                good = i < N and all(se_verify(pub, *world.signed[k][s]) for s in sel)
                if i >= N:   # Sybil device signs its own (off-feeder) windows
                    good = True
                Xs = nodes[k].train[sel]
                l_new = ae_scores(w + U[i], Xs).mean()
                l_old = ae_scores(w, Xs).mean()
                prov_ok = good and l_new <= 1.25 * l_old + 1e-3
            stats["pop_reject"] += int(not pop_ok)
            stats["prov_reject"] += int(pop_ok and not prov_ok)
            checks.append((i, idt, pop_ok, prov_ok))
        passed = [i for i, _, p, q in checks if p and q]
        ref = np.median(U[passed], 0) if passed else np.zeros(NPARAM)
        Uagg, Wagg = [], []               # pass 2: trust, key state, weights
        for i, idt, pop_ok, prov_ok in checks:
            cosv = float(U[i] @ ref / (np.linalg.norm(U[i]) * np.linalg.norm(ref) + 1e-12))
            s_i = float(pop_ok) * float(prov_ok) * max(0.0, cosv)
            if "trust" in ablate:
                if pop_ok and prov_ok:
                    Uagg.append(U[i]); Wagg.append(1.0)
                continue
            T[i] = lam * T[i] + (1 - lam) * s_i
            stt = ledger.state[idt.addr]
            if stt == ACTIVE and T[i] < tau_q:
                ledger.set_state(idt.addr, QUARANTINED); streak_q[i] = 0
            elif stt == QUARANTINED:
                streak_q[i] += 1
                streak_ok[i] = streak_ok[i] + 1 if s_i >= 0.6 else 0
                if streak_ok[i] >= 2:
                    ledger.set_state(idt.addr, ACTIVE); T[i] = max(T[i], tau_q + 0.05)
                elif streak_q[i] >= 3:
                    ledger.set_state(idt.addr, REVOKED)
            if ledger.state[idt.addr] == ACTIVE and pop_ok and prov_ok:
                Uagg.append(U[i]); Wagg.append(T[i])
        if Uagg:
            Ua, Wa = np.array(Uagg), np.array(Wagg)
            cap = np.median(np.linalg.norm(Ua, axis=1))
            Ua = Ua * np.minimum(1, cap / (np.linalg.norm(Ua, axis=1, keepdims=True) + 1e-12))
            if Wa.sum() > 0:
                if "wmedian" in ablate:      # plain trust-weighted mean
                    w = w + (Wa[:, None] * Ua).sum(0) / Wa.sum()
                else:                        # trust-weighted coordinate median
                    w = w + weighted_median(Ua, Wa)
        if track:
            hist.append((evaluate(w, nodes)[0],
                         np.mean([T[i] for i in range(N) if i not in bad]),
                         np.mean([T[i] for i in bad]) if bad else np.nan,
                         np.mean([T[i] for i in range(N, N + n_sybil)]) if n_sybil else np.nan))
    a, f1 = evaluate(w, nodes)
    out = {"auc": a, "f1": f1, "hist": hist, "stats": stats}
    if proposed:
        out["revoked"] = sum(1 for v in ledger.state.values() if v == REVOKED)
        out["log"] = ledger.log
    return out


# ---------------------------------------------------------------------------
# 6. Experiments, figures and tables
# ---------------------------------------------------------------------------
METHODS = ["FedAvg", "Multi-Krum", "Median", "Trimmed mean", "FLTrust",
           "PhysiTrust-FL"]
COL = 3.45  # IEEE column width in inches


def save_csv(name, header, rows):
    import csv
    with open(os.path.join(OUT, name), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(header)
        for r in rows:
            wr.writerow([str(x) for x in r])


def exp_pop_distribution(n_win=30, seed=11):
    rng = np.random.default_rng(seed)
    fd = Feeder(rng)
    D = electrical_distance()
    out = {"Honest": [], "Relocated (<2 ohm)": [], "Relocated (>=4 ohm)": [],
           "Replayed window": [], "Off-grid device": []}
    for _ in range(n_win):
        tr = pop_trial(fd, 120, 0.03, rng=rng)
        c, gV, gQ = tr["codes"], tr["gV"], tr["gQ"]
        for b in NODES:
            out["Honest"].append(pop_statistic(tr["Vm"][:, b], tr["Qf"][:, b - 1], c, gV[b], gQ[b - 1]))
            out["Replayed window"].append(pop_statistic(tr["Vr"][:, b], tr["Qr"][:, b - 1], c, gV[b], gQ[b - 1]))
            out["Off-grid device"].append(pop_statistic(tr["Vo"][:, b], tr["Qo"][:, b - 1], c, gV[b], gQ[b - 1]))
            for j in NODES:
                if j == b:
                    continue
                st = pop_statistic(tr["Vm"][:, j], tr["Qf"][:, j - 1], c, gV[b], gQ[b - 1])
                if D[b, j] < 2:
                    out["Relocated (<2 ohm)"].append(st)
                elif D[b, j] >= 4:
                    out["Relocated (>=4 ohm)"].append(st)
    rows = []
    for k, v in out.items():
        v = np.array(v)
        rows.append([k, len(v), np.mean(v < POP_THRESHOLD), np.median(v)])
    save_csv("table_pop_acceptance.csv", ["case", "trials", "accept_rate", "median_D2"], rows)
    fig, ax = plt.subplots(figsize=(COL, 1.9))
    data = [np.log10(np.array(v) + 1e-3) for v in out.values()]
    ax.boxplot(data, vert=False, widths=0.6, showfliers=False)
    ax.set_yticks(range(1, len(out) + 1))
    ax.set_yticklabels(list(out.keys()))
    ax.axvline(np.log10(POP_THRESHOLD), color="r", ls="--", lw=1)
    ax.text(np.log10(POP_THRESHOLD) + 0.08, 0.55, "accept threshold",
            color="r", fontsize=6.5)
    ax.set_xlabel(r"$\log_{10} D^2$ (fingerprint distance)")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_pop_distribution.png"), dpi=300)
    plt.close(fig)
    return rows


def max_voltage_impact(dq):
    P = np.repeat(P_NOM[None], 2, 0); Q = np.repeat(Q_NOM[None], 2, 0)
    Q[1, ACTUATORS] -= dq
    V, _ = power_flow(P, Q)
    return 100 * np.max(np.abs(np.abs(V[1]) - np.abs(V[0])))


def exp_pop_sweep(n_win=8, seed=12, dqs=(0.005, 0.01, 0.02, 0.03, 0.05),
                  Ls=(60, 120, 240)):
    rng = np.random.default_rng(seed)
    fd = Feeder(rng)
    rows = []
    for L in Ls:
        for dq in dqs:
            frr, far_o, far_r, far_l = [], [], [], []
            for _ in range(n_win):
                tr = pop_trial(fd, L, dq, rng=rng)
                c, gV, gQ = tr["codes"], tr["gV"], tr["gQ"]
                for b in NODES:
                    frr.append(pop_statistic(tr["Vm"][:, b], tr["Qf"][:, b - 1], c, gV[b], gQ[b - 1]) >= POP_THRESHOLD)
                    far_o.append(pop_statistic(tr["Vo"][:, b], tr["Qo"][:, b - 1], c, gV[b], gQ[b - 1]) < POP_THRESHOLD)
                    far_r.append(pop_statistic(tr["Vr"][:, b], tr["Qr"][:, b - 1], c, gV[b], gQ[b - 1]) < POP_THRESHOLD)
                    for j in NODES[::3]:
                        if j != b:
                            far_l.append(pop_statistic(tr["Vm"][:, j], tr["Qf"][:, j - 1], c, gV[b], gQ[b - 1]) < POP_THRESHOLD)
            rows.append([L, round(dq * 1000), np.mean(frr), np.mean(far_o),
                         np.mean(far_r), np.mean(far_l), max_voltage_impact(dq)])
    save_csv("table_pop_sweep.csv", ["L_s", "dq_kvar", "FRR", "FAR_offgrid",
                                     "FAR_replay", "FAR_relocation", "max_dV_pct"], rows)
    plot_pop_sweep(rows)
    return rows


def plot_pop_sweep(rows, Ls=(60, 120, 240)):
    R = np.array(rows, float)
    fig, ax = plt.subplots(figsize=(COL, 2.0))
    for L, mk in zip(Ls, "osd"):
        m = R[:, 0] == L
        ax.plot(R[m, 1], R[m, 5], marker=mk, ms=3, label=f"Relocation FAR, L={L} s")
    m = R[:, 0] == Ls[0]
    ax.plot(R[m, 1], R[m, 4], marker="x", ms=3, ls="--", label=f"Replay FAR, L={Ls[0]} s")
    ax.plot(R[m, 1], R[m, 2], marker="+", ms=3, ls="--", label=f"FRR, L={Ls[0]} s")
    ax.set_xlabel(r"Dither amplitude $\Delta q$ per inverter (kvar)")
    ax.set_ylabel("Rate"); ax.set_ylim(-0.01, 0.25)
    ax2 = ax.twinx()
    ax2.plot(R[m, 1], R[m, 6], color="k", lw=1, ls=":", label=r"max $|\Delta V|$ (right)")
    ax2.set_ylabel(r"max $|\Delta V|$ (%)")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    fig.legend(h1 + h2, l1 + l2, ncol=2, loc="upper center", fontsize=5.5, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.78))
    fig.savefig(os.path.join(OUT, "fig_pop_sweep.png"), dpi=300)
    plt.close(fig)


def exp_pop_actuators(n_win=6, seed=13):
    rng = np.random.default_rng(seed)
    fd = Feeder(rng)
    D = electrical_distance()
    sets = {"4 inverters": [18, 22, 25, 33], "6 inverters": [18, 22, 25, 33, 13, 29],
            "8 inverters": [18, 22, 25, 33, 9, 13, 29, 4]}
    bins = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 100)]
    rows, curves = [], {}
    for name, acts in sets.items():
        acts = np.array(acts) - 1
        th = pop_threshold(len(acts))
        acc = np.zeros((NB, NB)); frr = []
        for _ in range(n_win):
            tr = pop_trial(fd, 120, 0.03, acts=acts, rng=rng)
            c, gV, gQ = tr["codes"], tr["gV"], tr["gQ"]
            for b in NODES:
                frr.append(pop_statistic(tr["Vm"][:, b], tr["Qf"][:, b - 1], c, gV[b], gQ[b - 1]) >= th)
                for j in NODES:
                    if j != b:
                        acc[b, j] += pop_statistic(tr["Vm"][:, j], tr["Qf"][:, j - 1], c, gV[b], gQ[b - 1]) < th
        acc /= n_win
        pairs = np.array([(D[b, j], acc[b, j]) for b in NODES for j in NODES if j != b])
        curve = [pairs[(pairs[:, 0] >= lo) & (pairs[:, 0] < hi), 1].mean() for lo, hi in bins]
        curves[name] = curve
        rows.append([name, np.mean(frr), pairs[:, 1].mean()] + [round(x, 4) for x in curve])
    save_csv("table_pop_actuators.csv", ["set", "FRR", "FAR_all_pairs"] +
             [f"FAR_{lo}-{hi}ohm" for lo, hi in bins], rows)
    fig, ax = plt.subplots(figsize=(COL, 1.7))
    x = np.arange(len(bins)); wbar = 0.27
    for k, (name, cv) in enumerate(curves.items()):
        ax.bar(x + (k - 1) * wbar, cv, wbar, label=name)
    ax.set_xticks(x); ax.set_xticklabels(["0-1", "1-2", "2-3", "3-4", ">4"])
    ax.set_xlabel(r"Electrical distance between true and claimed bus ($\Omega$)")
    ax.set_ylabel("Relocation FAR"); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_pop_actuators.png"), dpi=300)
    plt.close(fig)
    return rows


def exp_fl_sweep(worlds, rounds, fracs=(0, .1, .2, .3, .4, .5, .6),
                 attacks=("SF", "GN", "ADP")):
    rows = []
    for att in attacks:
        for fr in fracs:
            for m in METHODS:
                res = [run_fl(wd, m, att, fr, 0, rounds, seed=s)
                       for s, wd in enumerate(worlds)]
                rows.append([att, fr, m, np.mean([r["auc"] for r in res]),
                             np.mean([r["f1"] for r in res])])
                print(f"  {att} f={fr:.1f} {m:14s} AUC={rows[-1][3]:.3f} F1={rows[-1][4]:.3f}", flush=True)
    if set(attacks) >= {"SF", "ADP"}:
        save_csv("table_fl_sweep.csv", ["attack", "fraction", "method", "AUC", "F1"], rows)
        plot_fl_sweep(rows)
    return rows


def plot_fl_sweep(rows, attacks=("SF", "ADP")):
    names = {"SF": "(a) Sign-flip x4", "GN": "Gaussian noise",
             "ADP": "(b) Adaptive data poisoning"}
    fig, axs = plt.subplots(1, len(attacks), figsize=(COL, 2.0), sharey=True)
    for ax, att in zip(axs, attacks):
        for m, mk in zip(METHODS, "o^sdvx"):
            pts = [(float(r[1]), float(r[4])) for r in rows if r[0] == att and r[2] == m]
            ax.plot(*zip(*pts), marker=mk, ms=2.5,
                    lw=1.4 if m == "PhysiTrust-FL" else 0.8, label=m)
        ax.set_title(names[att]); ax.set_xlabel("Compromised fraction")
    axs[0].set_ylabel("F1 of global detector")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, frameon=False, fontsize=5.5)
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    fig.savefig(os.path.join(OUT, "fig_fl_sweep.png"), dpi=300)
    plt.close(fig)


def exp_sybil(worlds, rounds, counts=(0, 16, 32, 48, 64), tag=""):
    rows = []
    for n in counts:
        for m in METHODS:
            res = [run_fl(wd, m, "SF", 0.0, n, rounds, seed=s) for s, wd in enumerate(worlds)]
            rows.append([n, m, np.mean([r["auc"] for r in res]), np.mean([r["f1"] for r in res])])
            print(f"  sybil={n} {m:14s} AUC={rows[-1][2]:.3f} F1={rows[-1][3]:.3f}", flush=True)
    save_csv(f"table_sybil{tag}.csv", ["sybils", "method", "AUC", "F1"], rows)
    if tag:
        return rows
    plot_sybil(rows)
    return rows


def plot_sybil(rows):
    fig, ax = plt.subplots(figsize=(COL, 1.8))
    for m, mk in zip(METHODS, "o^sdvx"):
        pts = [(int(r[0]), float(r[3])) for r in rows if r[1] == m]
        ax.plot(*zip(*pts), marker=mk, ms=3, label=m)
    ax.set_xlabel("Credentialed Sybil devices (32 genuine nodes)")
    ax.set_ylabel("F1 of global detector")
    fig.legend(*ax.get_legend_handles_labels(), loc="upper center", ncol=3, frameon=False, fontsize=5.5)
    fig.tight_layout(rect=(0, 0, 1, 0.84)); fig.savefig(os.path.join(OUT, "fig_sybil.png"), dpi=300)
    plt.close(fig)
    return rows


def exp_ablation(worlds, rounds, which=None, tag=""):
    cases = [("Full PhysiTrust-FL", ()), ("without PoP", ("pop",)),
             ("without provenance check", ("prov",)), ("without trust state", ("trust",)),
             ("weighted mean instead of median", ("wmedian",))]
    scen = [("SF", 0.6, 0), ("ADP", 0.0, 48), ("ADP", 0.3, 0)]
    rows = []
    for ci, (name, ab) in enumerate(cases):
        if which is not None and ci not in which:
            continue
        row = [name]
        for att, fr, ns in scen:
            res = [run_fl(wd, "PhysiTrust-FL", att, fr, ns, rounds, seed=s, ablate=ab)
                   for s, wd in enumerate(worlds)]
            row.append(round(np.mean([r["f1"] for r in res]), 3))
        rows.append(row)
        print("  ablation", row, flush=True)
    save_csv(f"table_ablation{tag}.csv", ["variant", "F1_SF60", "F1_ADPsybil48",
                                    "F1_ADP30"], rows)
    return rows


def exp_trust(world, rounds):
    r = run_fl(world, "PhysiTrust-FL", "SF", 0.25, 16, rounds, seed=0, track=True)
    H = np.array(r["hist"], float)
    fig, ax = plt.subplots(figsize=(COL, 1.8))
    x = np.arange(1, len(H) + 1)
    ax.plot(x, H[:, 1], label="Honest nodes (mean T)")
    ax.plot(x, H[:, 2], lw=2.2, alpha=0.6, label="Compromised nodes, sign-flip")
    ax.plot(x, H[:, 3], ls="--", label="Sybil devices (off-feeder)")
    ax.plot(x, H[:, 0], "k--", lw=0.9, label="Global AUC")
    ax.axhline(0.3, color="r", lw=0.6, ls=":")
    ax.set_xlabel("FL round"); ax.set_ylabel("Trust / AUC"); ax.set_ylim(-0.02, 1.02)
    ax.legend(frameon=False, fontsize=5.5, loc="center right")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_trust.png"), dpi=300)
    plt.close(fig)
    first = {}
    for rnd, addr, st in r["log"]:
        first.setdefault(st, rnd)
    save_csv("table_trust_events.csv", ["state", "first_round"], list(first.items()))
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="1 seed, 15 rounds")
    ap.add_argument("--stage", default="all",
                    choices=["all", "crypto", "pop", "fl", "sybil", "ablation", "trust"],
                    help="run one stage only (useful on slow machines)")
    ap.add_argument("--attack", default=None, help="fl stage: SF, GN or ADP only")
    ap.add_argument("--fracs", default=None, help="fl stage: e.g. 0,0.1,0.2")
    ap.add_argument("--counts", default=None,
                    help="sybil stage: Sybil counts, e.g. 0,16; ablation stage: variant indices, e.g. 0,1")
    ap.add_argument("--tag", default="", help="suffix for CSV names of partial runs")
    ap.add_argument("--replot", action="store_true",
                    help="redraw the FL sweep figure from the saved CSV files")
    a = ap.parse_args()
    seeds = [0] if a.quick else [0, 1]
    rounds = 15 if a.quick else 25
    st = a.stage
    t0 = time.time()
    if a.replot:
        import csv, glob
        rows = []
        for fn in sorted(glob.glob(os.path.join(OUT, "table_fl_sweep_*.csv"))):
            with open(fn) as f:
                rows += list(csv.reader(f))[1:]
        plot_fl_sweep(rows)
        with open(os.path.join(OUT, "table_pop_sweep.csv")) as f:
            plot_pop_sweep(list(csv.reader(f))[1:])
        srows = []
        for fn in sorted(glob.glob(os.path.join(OUT, "table_sybil_*.csv"))):
            with open(fn) as f:
                srows += list(csv.reader(f))[1:]
        if srows:
            srows.sort(key=lambda r: int(r[0]))
            save_csv("table_sybil.csv", ["sybils", "method", "AUC", "F1"], srows)
            plot_sybil(srows)
        arows = []
        for fn in sorted(glob.glob(os.path.join(OUT, "table_ablation_*.csv"))):
            with open(fn) as f:
                arows += list(csv.reader(f))[1:]
        if arows:
            save_csv("table_ablation.csv", ["variant", "F1_SF60", "F1_ADPsybil48", "F1_ADP30"], arows)
        return
    if st in ("all", "crypto"):
        cb = benchmark_crypto(50 if a.quick else 200)
        for k, v in cb.items():
            print(f"  {k:32s} {v:8.3f} ms")
        save_csv("table_crypto_ms.csv", ["operation", "ms"], [[k, round(v, 4)] for k, v in cb.items()])
        save_csv("table_gas_estimate.csv", ["call", "gas"], list(gas_estimate().items()))
    if st in ("all", "pop"):
        print(exp_pop_distribution(10 if a.quick else 30))
        print(exp_pop_sweep(3 if a.quick else 8))
        print(exp_pop_actuators(2 if a.quick else 6))
    if st in ("all", "fl", "sybil", "ablation", "trust"):
        worlds = [World(s) for s in seeds]
        if st in ("all", "fl"):
            atts = (a.attack,) if a.attack else ("SF", "GN", "ADP")
            rows = []
            fr = tuple(float(x) for x in a.fracs.split(",")) if a.fracs else (0, .1, .2, .3, .4, .5, .6)
            for att in atts:
                rows += exp_fl_sweep(worlds, rounds, fracs=fr, attacks=(att,))
                save_csv(f"table_fl_sweep_{att}{a.tag}.csv",
                         ["attack", "fraction", "method", "AUC", "F1"],
                         [r for r in rows if r[0] == att])
        if st in ("all", "sybil"):
            cn = tuple(int(x) for x in a.counts.split(",")) if a.counts else (0, 16, 32, 48, 64)
            exp_sybil(worlds, rounds, counts=cn, tag=a.tag)
        if st in ("all", "ablation"):
            wh = [int(x) for x in a.counts.split(",")] if a.counts else None
            exp_ablation(worlds, rounds, which=wh, tag=a.tag)
        if st in ("all", "trust"):
            exp_trust(worlds[0], rounds)
    print(f"Done in {(time.time() - t0) / 60:.1f} min. Results in ./{OUT}")


if __name__ == "__main__":
    main()
