# PhysiTrust-FL

**Grid-Challenged Proof-of-Presence and Hardware-Bound Measurement Provenance for Trust Management in Federated Grid-Edge Anomaly Detection**

![Python](https://img.shields.io/badge/python-3.9%2B-blue) ![Status](https://img.shields.io/badge/status-research%20prototype-orange)

PhysiTrust-FL is a Web3 public key infrastructure (PKI) for smart-grid devices that train a shared anomaly detector with federated learning. A normal digital signature proves *who* sent a model update. PhysiTrust-FL also proves *where* the sender is on the grid and *that its training data was really measured*, by using the power grid itself as a challenge.

This repository contains the full simulation used in the course paper (Public Key Infrastructure and Trust Management, VIT Vellore). It reproduces every figure and table in the paper.

---

## Contents

- [The problem](#the-problem)
- [How it works](#how-it-works)
- [Repository structure](#repository-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [How the code is organised](#how-the-code-is-organised)
- [Limitations](#limitations)
- [Intellectual property notice](#intellectual-property-notice)
- [Author](#author)

---

## The problem

Smart meters and line monitors on a distribution feeder can train a shared anomaly detector with federated learning (FL). Only model updates leave each device, so customer data stays private. But the aggregator cannot tell whether an update really comes from a device installed on the feeder.

- **Statistical defences** (Krum, median, trimmed mean) assume honest devices are the majority. An attacker who adds identities breaks that assumption.
- **Blockchain FL and decentralized identifiers (DIDs)** prove who signed an update, not where the signer is.
- A real meter that was bought, credentialed and then kept on a desk at home signs updates just as well as one on the feeder.

## How it works

![Architecture](docs/figures/fig_architecture.png)

Each FL round adds two physical checks to the usual identity check.

### 1. Grid-challenged proof-of-presence (PoP)

1. The utility commits to a secret seed on the ledger by publishing only its hash.
2. Four smart inverters change their reactive power by ±30 kvar in a pseudo-random pattern derived from the seed.
3. Every edge node signs and commits its measurements before the seed is revealed.
4. The verifier estimates each node's response gains (voltage and branch Q-flow) by least squares and compares them with the expected fingerprint of the node's registered bus using a chi-square test.

A device that is not on the feeder sees no ripple. A replayed window carries an old pattern. A relocated device shows the fingerprint of the wrong bus. The ripple changes bus voltage by less than 0.24%.

![IEEE 33-bus feeder](docs/figures/fig_feeder.png)

### 2. Hardware-bound measurement provenance

A secure element (P-256) in the metering path signs every raw measurement window. The training set is committed as a Merkle root inside the EIP-712 signed update. Each round the verifier spot-checks four random signed windows and rejects a model that performs much worse on the node's own real data. The attacker cannot defend a poisoned model with fabricated data, because fabricated data has no valid hardware signature.

### 3. Trust and key states

A trust score combines both checks with the update's agreement with the median of nodes that passed. Keys move between `ACTIVE`, `QUARANTINED` and `REVOKED` on chain. Accepted updates are combined with a trust-weighted coordinate-wise median.

### One round

| Step | Action |
| --- | --- |
| 1 | Aggregator posts H(w_r); utility posts keccak(seed) |
| 2 | Inverters apply the dither derived from the seed for L seconds |
| 3 | Each secure element signs its raw window; node commits the digest |
| 4 | Utility reveals the seed; contract checks it against the commit |
| 5 | Node trains locally and submits an EIP-712 signed update |
| 6 | Contract checks signature, key state, nonce and commit order; verifier runs PoP and provenance |
| 7 | Trust scores and key states are updated on chain |
| 8 | Trust-weighted median gives w_(r+1); its hash is anchored and verified by nodes |

## Repository structure

```
physitrust_sim/
├── physitrust_sim.py        # complete simulation (power flow, PoP, PKI, FL, experiments)
├── requirements.txt
├── LICENSE                  # all rights reserved (patent pending consideration)
├── results/                 # figures (PNG) and tables (CSV) from the full run
└── docs/
    ├── PhysiTrust-FL_Course_Paper.pdf
    └── figures/             # architecture and feeder diagrams
```

## Installation

Requires Python 3.9 or newer.

```bash
git clone https://github.com/manoj291205/physitrust_sim.git
cd physitrust_sim
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

```bash
python physitrust_sim.py --quick                 # 1 seed, 15 rounds, about 3 minutes
python physitrust_sim.py                         # 2 seeds, 25 rounds, about 30 minutes
```

Run one stage at a time on slower machines:

```bash
python physitrust_sim.py --stage crypto          # signing / verification timings and gas estimates
python physitrust_sim.py --stage pop             # proof-of-presence experiments (under 1 minute)
python physitrust_sim.py --stage fl --attack SF  # FL sweep for one attack: SF, GN or ADP
python physitrust_sim.py --stage sybil
python physitrust_sim.py --stage ablation
python physitrust_sim.py --stage trust
python physitrust_sim.py --replot                # rebuild combined figures from saved CSVs
```

Partial runs can be split further with `--fracs 0,0.1,0.2 --tag _a` (FL stage) or `--counts 0,16 --tag _a` (Sybil and ablation stages), then merged with `--replot`.

All outputs are written to `./results`.

## Results

All numbers are averages over 2 random seeds and 25 FL rounds on the IEEE 33-bus feeder with 32 edge nodes.

### Proof-of-presence

| Case | Accepted |
| --- | --- |
| Honest node | 99.9% |
| Off-grid device | 0% |
| Replayed window | 0% |
| Device moved ≥ 4 Ω away | 0% |
| Device moved < 2 Ω away | 42.4% (location resolution limit) |

![PoP distribution](results/fig_pop_distribution.png)

### Robustness of the global detector (F1)

| Method | Sign-flip 60% | Gaussian noise 60% | Adaptive poisoning 30% | 64 credentialed Sybils |
| --- | --- | --- | --- | --- |
| FedAvg | 0.05 | 0.81 | 0.81 | 0.72 |
| Multi-Krum | 0.06 | 0.86 | 0.88 | 0.67 |
| Median | 0.08 | 0.88 | 0.87 | 0.68 |
| Trimmed mean | 0.07 | 0.85 | 0.83 | 0.72 |
| FLTrust | 0.67 | 0.66 | 0.65 | 0.63 |
| **PhysiTrust-FL** | **0.87** | **0.89** | **0.88** | **0.89** |

![FL sweep](results/fig_fl_sweep.png)
![Sybil devices](results/fig_sybil.png)

### Ablation (F1)

| Variant | Sign-flip 60% | 48 Sybils | Adaptive poisoning 30% |
| --- | --- | --- | --- |
| Full PhysiTrust-FL | 0.867 | 0.889 | 0.876 |
| Without proof-of-presence | 0.867 | 0.649 | 0.876 |
| Without provenance check | 0.029 | 0.889 | 0.874 |
| Without trust state | 0.850 | 0.888 | 0.865 |
| Weighted mean instead of median | 0.877 | 0.881 | 0.842 |

### Overhead

| Operation | Cost |
| --- | --- |
| Secure element P-256 sign (one window) | 0.08 ms |
| EIP-712 secp256k1 sign (Python) | 4.8 ms |
| ecrecover verify (Python) | 6.8 ms |
| commitMeasurement + submitUpdate | about 118k gas per node per round (estimated) |

## How the code is organised

`physitrust_sim.py` has six numbered sections.

| Section | Contents |
| --- | --- |
| 1. Feeder | IEEE 33-bus data (Baran and Wu, 1989), vectorised backward/forward sweep power flow, electrical distance |
| 2. Proof-of-presence | Chip generation from the seed, dithered measurement windows, expected gains, least-squares gain estimation, chi-square test |
| 3. Web3 PKI | Merkle roots, secure element (P-256), EIP-712 typed updates, edge identities, a Python mirror of the smart-contract logic, crypto benchmark, gas estimate |
| 4. Telemetry and model | Synthetic 1-minute telemetry, six anomaly types, per-node datasets, NumPy autoencoder (40-12-40) |
| 5. Aggregation and protocol | FedAvg, Multi-Krum, median, trimmed mean, FLTrust, weighted median, attacks (SF, GN, DP, ADP, Sybil), the PhysiTrust-FL round in `run_fl()` |
| 6. Experiments | Functions that produce every figure and table, plus the command-line interface |

## Limitations

- All data is simulated; a field test on a real feeder has not been done.
- The smart-contract logic is emulated in Python. Signatures are real (eth-account, cryptography), but gas figures are estimated from the EVM fee schedule, not measured on a chain.
- Buses less than about 2 Ω apart can be confused by the presence test.
- Adaptive data poisoning that stays within the loss gate is bounded, not removed; with half or more of the genuine nodes compromised, PhysiTrust-FL falls below plain averaging.
- The scheme needs controllable inverters and an accurate feeder model held by the utility.

## Intellectual property notice

The PhysiTrust-FL mechanism is under consideration for a patent application. Copyright © 2026 Manojkumar Ashokkumar. All rights reserved. No license is granted to use, copy, modify or distribute this code or the described method without written permission from the author.

## Author

**Manojkumar Ashokkumar**
B.Tech Computer Science and Engineering (Blockchain), Vellore Institute of Technology, Vellore, India
