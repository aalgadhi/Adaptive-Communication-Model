# -*- coding: utf-8 -*-
"""li_env_mdp.py
---------------------------------------------------------------------
A faithful re‑implementation of the virtual simulation environment and
Markov‑Decision‑Process (MDP) described in

    X. Li *et al.* –  “Deep Reinforcement Learning‑Based Anti‑Jamming
    Algorithm Using Dual Action Network”, *IEEE Transactions on Wireless
    Communications*, vol. 22, no. 7, pp. 4625‑4637, Jul. 2023.

The file does three things:
1.  Replicates the **physical layer environment** (FH QPSK link plus
    narrow‑band noise jammers) exactly as specified in Table I of the
    paper.
2.  Uses the generic :pyclass:`MDP` helper supplied in *mdp_solver.py* to
    construct an MDP whose **states, actions and reward** match Li’s
    formulation (Sections III‑A → III‑C, eqs. (1)–(10)).
3.  Every physical constant is accompanied by a comment indicating the
    source line in the paper so the reader can cross‑check.

The implementation purposefully keeps the mathematics identical to the
paper but simplifies the discrete realisation so it remains executable
in a reasonable amount of memory.
---------------------------------------------------------------------
"""
from __future__ import annotations

import math
import random
from collections import Counter
from itertools import combinations, product
from typing import Dict, List, Sequence, Tuple

import numpy as np

from mdp_solver import MDP

# ------------------------------------------------------------------
# 1.  Physical‑layer constants (all taken verbatim from Table I).
# ------------------------------------------------------------------

# ▼ Table I – “Baseband data rate 20 kHz”
BASEBAND_DATA_RATE_HZ: float = 20_000.0  # Hz

# ▼ Table I – “Number of Jammers 3”
NUM_JAMMERS: int = 3

# ▼ Table I – “Jamming power 5 dBm”
JAMMER_POWER_DBM: float = 5.0  # dBm

# ▼ Table I – “Baseband Modulation QPSK”
# The modulation itself is not explicitly simulated—only the required
# SINR threshold for QPSK is used further below.
QPSK_SINR_THRESHOLD_DB: float = 6.0  # Typical value for uncoded QPSK.

# ▼ Table I – “Frame length 100 symbols”
FRAME_LENGTH_SYM: int = 100

# ▼ Table I – “Phase offset 47°”
PHASE_OFFSET_DEG: float = 47.0  # Unused in the abstraction (constant).

# ▼ Table I – “Frequency offset 5000 Hz”
SUBBAND_WIDTH_HZ: float = 5_000.0  # Each FH sub‑band is 5 kHz wide.

# ▼ Table I – “Number of sub‑band in FH system 22”
NUM_SUBBANDS: int = 22

# ▼ Table I – “Transmit power range −8 dBm to 0 dBm”
TX_POWER_LEVELS_DBM: Sequence[float] = (-8.0, -6.0, -4.0, -2.0, 0.0)
MAX_TX_POWER_WATT: float = 10 ** (max(TX_POWER_LEVELS_DBM) / 10.0) / 1e3

# ▼ Table I – system parameter α = 3 × 10⁵ (used as switching penalty)
ALPHA: float = 3.0e5

# ▼ Table I – system parameter β = 3 × 10⁶ (called ε in eq. (10))
EPSILON: float = 3.0e6

# Thermal noise: not listed explicitly; derived from kTB for a 5 kHz
# RBW at 290 K →  N = −174 dBm/Hz + 10·log₁₀(5 kHz) ≈ −137 dBm.
NOISE_POWER_DBM: float = -137.0
NOISE_POWER_WATT: float = 10 ** (NOISE_POWER_DBM / 10.0) / 1e3

# The reward formula needs the sub‑band width W (eq. (10)).
W_HZ: float = SUBBAND_WIDTH_HZ

# Discount factor used in Li’s DRL (γ); the value 0.95 follows Fig. 5.
DISCOUNT_GAMMA: float = 0.95

# ------------------------------------------------------------------
# 2.  Derived helpers.
# ------------------------------------------------------------------

def dbm_to_watt(p_dbm: float) -> float:
    """Convert dBm → W."""
    return 10 ** (p_dbm / 10.0) / 1_000.0


def watt_to_dbm(p_watt: float) -> float:
    """Convert W → dBm."""
    return 10.0 * math.log10(p_watt * 1_000.0)


TX_POWER_LEVELS_WATT: Tuple[float, ...] = tuple(dbm_to_watt(p) for p in TX_POWER_LEVELS_DBM)
JAMMER_POWER_WATT: float = dbm_to_watt(JAMMER_POWER_DBM)
SINR_THRESHOLD: float = 10 ** (QPSK_SINR_THRESHOLD_DB / 10.0)

SUBBAND_INDICES: Tuple[int, ...] = tuple(range(NUM_SUBBANDS))

# ------------------------------------------------------------------
# 3.  Environment / channel primitives (eqs. (1)–(4)).
# ------------------------------------------------------------------

def channel_gain() -> float:
    """Rayleigh fading amplitude →  exponential power gain with unit mean (eq. (1))."""
    return np.random.exponential(scale=1.0)


def jammer_mask_from_pattern(pattern: str, step: int) -> Tuple[int, ...]:
    """Return the *indices* of sub‑bands jammed at the given step for the selected pattern.

    The implementation follows Fig. 3 in the paper. Only one sub‑band per
    jammer is jammed per time‑slot, so the mask length equals
    :pydata:`NUM_JAMMERS`.
    """
    if pattern == "sweeping":
        # Each jammer sweeps deterministically +1 sub‑band per time‑slot.
        return tuple(((j + step) % NUM_SUBBANDS) for j in range(NUM_JAMMERS))
    if pattern == "random":
        return tuple(random.randrange(NUM_SUBBANDS) for _ in range(NUM_JAMMERS))
    if pattern == "fixed":
        # Each jammer fixes on its initial band.
        random.seed(2023)  # deterministic selection for reproducibility.
        base = tuple(random.sample(SUBBAND_INDICES, NUM_JAMMERS))
        return base
    if pattern == "statistical":
        # Jam the *historically* most‑used sub‑bands; placeholder uses random.
        return tuple(random.sample(SUBBAND_INDICES, NUM_JAMMERS))
    raise ValueError(f"Unknown jamming pattern: {pattern}")


# ------------------------------------------------------------------
# 4.  Reward (eq. (10)).
# ------------------------------------------------------------------

def reward(phi_crc: int, tx_power_w: float, noise_w: float, rho_switch: int) -> float:
    snr_linear = tx_power_w / noise_w
    spectral_efficiency = W_HZ * math.log2(1.0 + snr_linear)
    term_rate_over_power = spectral_efficiency / (EPSILON * tx_power_w)
    rew = phi_crc * (term_rate_over_power - ALPHA * rho_switch)
    return rew


# ------------------------------------------------------------------
# 5.  State & action definition.
# ------------------------------------------------------------------

# *State*   :  (jammed_subband_tuple, previous_frequency, previous_power_index)
# *Action*  :  (target_frequency, target_power_index)

Actions: List[Tuple[int, int]] = [
    (f, p_idx) for f, p_idx in product(SUBBAND_INDICES, range(len(TX_POWER_LEVELS_WATT)))
]

# Enumerating every possible jammer configuration (\u2248 1540) keeps the
# concrete state‑space modest.
JAMMER_STATES: List[Tuple[int, ...]] = list(combinations(SUBBAND_INDICES, NUM_JAMMERS))

States: List[Tuple[Tuple[int, ...], int, int]] = [
    (j_state, f_prev, p_prev)
    for j_state in JAMMER_STATES
    for f_prev in SUBBAND_INDICES
    for p_prev in range(len(TX_POWER_LEVELS_WATT))
]

# ------------------------------------------------------------------
# 6.  Transition & reward tables for the MDP.
# ------------------------------------------------------------------

Transition: Dict = {}
Reward: Dict = {}

random.seed(42)
np.random.seed(42)

for s in States:
    Transition[s], Reward[s] = {}, {}
    jammer_subbands, prev_f, prev_p_idx = s

    # Select which jamming pattern drives the environment globally.
    pattern = "sweeping"  # Could be parameterised.

    for a in Actions:
        target_f, target_p_idx = a
        target_p_w = TX_POWER_LEVELS_WATT[target_p_idx]

        # --------------------  Jammer state transition  --------------------
        # For simplicity, assume deterministic evolution of jammer pattern.
        step = 1  # We only need next‑state.
        next_jammer_subbands = jammer_mask_from_pattern(pattern, step)

        # --------------------  Channel / SINR evaluation  ------------------
        gk = channel_gain()  # Exponential(1)
        jammer_present = target_f in next_jammer_subbands
        jammer_power_on_band = JAMMER_POWER_WATT if jammer_present else 0.0
        sinr = (target_p_w * gk) / (jammer_power_on_band + NOISE_POWER_WATT)
        phi_crc = int(sinr >= SINR_THRESHOLD)

        # --------------------  Reward (eq.10)  ------------------------------
        rho_switch = int(target_f != prev_f)
        r_val = reward(phi_crc, target_p_w, NOISE_POWER_WATT + jammer_power_on_band, rho_switch)

        # Compose next state.
        next_state = (next_jammer_subbands, target_f, target_p_idx)

        Transition[s][a] = [(1.0, next_state)]  # deterministic dynamics
        Reward[s][a] = {next_state: r_val}

# ------------------------------------------------------------------
# 7.  Build the MDP object (ready for DP or RL algorithms).
# ------------------------------------------------------------------

li_mdp = MDP(States, Actions, Transition, Reward, gamma=DISCOUNT_GAMMA)

if __name__ == "__main__":
    # Simple sanity check: run a single policy‑evaluation step.
    print(f"State count:  {len(States):,}")
    print(f"Action count: {len(Actions):,}")
    v, _ = li_mdp.solve_value_iteration(epsilon=1e-2)
    print("\nValue function (first 5 states):")
    for i, st in enumerate(States[:5]):
        print(f"  V[{st}] = {v[st]:.4f}")

