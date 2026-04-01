import math
import random

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from mdp_solver import MDP

# Random seed
SEED = 42

# Li-inspired anti-jamming setup
CHANNELS = [1, 2, 3]
JAMMING_VECTORS = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]

POWER_LEVELS_DBM = [-9.5, -8.5, -7.5] # Taken from Li's work
POWER_LEVELS = [1e-3 * (10 ** (p_dbm / 10.0)) for p_dbm in POWER_LEVELS_DBM]
DBM_FROM_POWER = {p: p_dbm for p, p_dbm in zip(POWER_LEVELS, POWER_LEVELS_DBM)}

# Independent per-channel quality states. Kept coarse for tractability.
SNR_STATES = [0, 1, 2]
CHANNEL_SNR_DB = {
    0: -4.0,
    1: 1.0,
    2: 6.0,
    3: 11
}

# -----------------------------
# Arsalan-style AMC ingredients
# -----------------------------
# Six AMC choices M = 2^k, k = 1..6.
MCS_LEVELS = [1, 2, 3, 4, 5]
MCS_ORDER = {
    1: 2,
    2: 4,
    3: 8,
    4: 16,
    5: 32
}

# Rate r(i) = log2(M) bits/symbol scaled by a base bandwidth.
BASE_DATA_RATE_HZ = 20e3
MCS_RATE = {m: BASE_DATA_RATE_HZ * math.log2(MCS_ORDER[m]) for m in MCS_LEVELS}

# Arsalan uses an empirical/state-wise mapping (state, MCS) -> FER.
# We keep that structure, but on a compressed 4-state channel abstraction to
# keep the hybrid Li+Arsalan model solvable.
# Rows = channel state 0..3 (worst -> best), columns = MCS 1..6.
FER_TABLE = {
    0: {1: 0.92, 2: 0.97, 3: 0.992, 4: 0.998, 5: 0.9995, 6: 0.9998},
    1: {1: 0.28, 2: 0.48, 3: 0.72, 4: 0.88, 5: 0.95, 6: 0.985},
    2: {1: 0.02, 2: 0.06, 3: 0.14, 4: 0.28, 5: 0.48, 6: 0.70},
    3: {1: 1e-4, 2: 8e-4, 3: 0.004, 4: 0.02, 5: 0.08, 6: 0.22},
}

# Arsalan-style switching cost, using the quadratic example structure.
MCS_SWITCH_COST_COEFF = 0.35

# Li-style frequency switching overhead and power penalty.
FREQ_SWITCH_COST = 0.6
POWER_COST_COEFF = 0.15

# Jammer and power effects are folded into an effective discrete channel state,
# then FER is looked up exactly from (effective_state, MCS).
JAMMER_STATE_SHIFT = -5
POWER_STATE_SHIFT = {
    -9.5: -1,
    -8.5: 0,
    -7.5: 1,
}
CRC_STATE_THRESHOLD = 1


def channel_transition_probs(curr_idx):
    """Independent 4-state birth-death FSMC per channel."""
    if curr_idx == 0:
        return [(0.65, 0), (0.35, 1)]
    if curr_idx == max(SNR_STATES):
        return [(0.35, curr_idx - 1), (0.65, curr_idx)]
    return [(0.175, curr_idx - 1), (0.65, curr_idx), (0.175, curr_idx + 1)]


def independent_channel_outcomes(curr_channel_state_tuple):
    """Cartesian product of the three independent channel FSMCs."""
    outcomes = [(1.0, ())]
    for curr_idx in curr_channel_state_tuple:
        next_outcomes = []
        for base_prob, base_state in outcomes:
            for p, next_idx in channel_transition_probs(curr_idx):
                next_outcomes.append((base_prob * p, base_state + (next_idx,)))
        outcomes = next_outcomes
    return outcomes


def effective_channel_state(jammer_state, channel_state_tuple, action):
    """
    Build an effective discrete link-quality state, then use Arsalan-style
    table lookup from that state and the chosen MCS.
    """
    target_f, _, target_p = action
    chosen_state = channel_state_tuple[target_f - 1]
    power_dbm = DBM_FROM_POWER[target_p]

    eff_state = chosen_state + POWER_STATE_SHIFT[power_dbm]
    if jammer_state[target_f - 1] == 1:
        eff_state += JAMMER_STATE_SHIFT

    return int(np.clip(eff_state, min(SNR_STATES), max(SNR_STATES)))


def fer_from_state_and_mcs(state_idx, mcs_level):
    """Arsalan-style one-to-one mapping: (state, MCS) -> FER."""
    fer = FER_TABLE[state_idx][mcs_level]
    # return min(max(float(fer), 1e-6), 1.0 - 1e-6)
    return fer


def arsalan_delay_cost(state_idx, mcs_level):
    """d_i(s) = 1 / (r(i) * (1 - FER(s, i)))."""
    fer = fer_from_state_and_mcs(state_idx, mcs_level)
    expected_good_rate = max(MCS_RATE[mcs_level] * (1.0 - fer), 1e-9)
    return 1.0 / expected_good_rate


def link_metrics(jammer_state, channel_state_tuple, action):
    target_f, target_mcs, _ = action
    raw_quality_state = channel_state_tuple[target_f - 1]
    eff_state = effective_channel_state(jammer_state, channel_state_tuple, action)
    fer = fer_from_state_and_mcs(eff_state, target_mcs)
    success_prob = 1.0 - fer
    delay_cost = arsalan_delay_cost(eff_state, target_mcs)
    crc_ok = 1.0 if eff_state >= CRC_STATE_THRESHOLD else 0.0
    
    return {
        "raw_quality_state": raw_quality_state,
        "effective_quality_state": eff_state,
        "fer": fer,
        "success_prob": success_prob,
        "delay_cost": delay_cost,
        "crc_ok": crc_ok,
        "rate": MCS_RATE[target_mcs],
        "goodput": MCS_RATE[target_mcs] * success_prob,
    }


def comm_mdp_reward(state, next_state, action):
    """
    Hybrid reward:
    - Delay and MCS switching follow Arsalan's formulation.
    - Frequency switching and power penalties follow Li's design goals.
    """
    curr_jammer_state, curr_channel_state_tuple, curr_mcs, curr_f = state
    next_jammer_state, next_channel_state_tuple, next_mcs, next_f = next_state
    target_f, target_mcs, target_p = action

    metrics = link_metrics(curr_jammer_state, curr_channel_state_tuple, action)

    delay_cost = metrics["delay_cost"]
    mcs_switch_cost = MCS_SWITCH_COST_COEFF * ((target_mcs - curr_mcs) ** 2)
    freq_switch_cost = FREQ_SWITCH_COST if target_f != curr_f else 0.0
    power_cost = POWER_COST_COEFF * (target_p / max(POWER_LEVELS))

    total_cost = math.log(1e4 * delay_cost) + mcs_switch_cost + freq_switch_cost + power_cost + (next_jammer_state[target_f-1] == 1)

    def print_reward_info():
        print(
            f"""
            delay * 1e4:    {delay_cost * 1e4}
            delay log:      {math.log(delay_cost * 1e4)}
            mcs_sc:         {mcs_switch_cost}
            freq_sc:        {freq_switch_cost}
            power_c:        {power_cost}
            jammer:         {5 * (next_jammer_state[target_f-1] == 1)}
            total_cost:     {-total_cost}
            """
        )

    # print_reward_info()

    return -total_cost


def build_comm_mdp(jammer="reactive"):
    actions = [(f, m, p) for f in CHANNELS for m in MCS_LEVELS for p in POWER_LEVELS]

    channel_state_tuples = [
        (c1, c2, c3)
        for c1 in SNR_STATES
        for c2 in SNR_STATES
        for c3 in SNR_STATES
    ]

    states = [
        (J, C_tuple, mcs, freq)
        for J in JAMMING_VECTORS
        for C_tuple in channel_state_tuples
        for mcs in MCS_LEVELS
        for freq in CHANNELS
    ]

    T, R = {}, {}
    random.seed(SEED)

    for s in states:
        T[s], R[s] = {}, {}
        curr_J, curr_C_tuple, _, _ = s

        for a in actions:
            target_f, _, target_p = a

            if jammer == "sweeper":
                next_jammer_idx = (JAMMING_VECTORS.index(curr_J) + 1) % len(JAMMING_VECTORS)
                next_jammer_vectors = [JAMMING_VECTORS[next_jammer_idx]]
                jammer_probs = [1.0]
                
            elif jammer == "reactive":
                p_follow = 0.55 + 0.30 * (target_p / max(POWER_LEVELS))
                p_follow = min(0.90, max(0.55, p_follow))
                jammer_probs = [(1.0 - p_follow) / 2.0] * 3
                jammer_probs[target_f - 1] = p_follow
                next_jammer_vectors = JAMMING_VECTORS
            
            else:
                next_jammer_vectors = JAMMING_VECTORS
                jammer_probs = [1 / 3, 1 / 3, 1 / 3]

            channel_outcomes = independent_channel_outcomes(curr_C_tuple)
            outcomes = []
            for pJ, next_J in zip(jammer_probs, next_jammer_vectors):
                for pC, next_C_tuple in channel_outcomes:
                    prob = pJ * pC
                    next_s = (next_J, next_C_tuple, a[1], a[0])
                    outcomes.append((prob, next_s))

            T[s][a] = outcomes
            R[s][a] = {}
            for _, next_s in outcomes:
                reward = comm_mdp_reward(s, next_s, a)
                R[s][a][next_s] = reward

    return MDP(states, actions, T, R), states

def representative_state(prev_mcs=1, prev_f=1, jammer=(0, 1, 0), channel_tuple=(0, 3, 6)):
    return (jammer, channel_tuple, prev_mcs, prev_f)

def run_comparison(mdp, policy, greedy):
    seed = 227
    random.seed(seed)
    start_state = random.choice(mdp.states)
    horizon = 3000
    num_episodes = 10

    res_opt = mdp.simulate_monte_carlo(start_state, num_episodes=num_episodes, horizon=horizon, policy=policy, seed=seed)
    res_grd = mdp.simulate_monte_carlo(start_state, num_episodes=num_episodes, horizon=horizon, policy=greedy, seed=seed)
    res_rnd = mdp.simulate_monte_carlo(start_state, num_episodes=num_episodes, horizon=horizon, policy=None, seed=seed)

    avg_opt = [np.mean([ep["trajectory"][j][3] for ep in res_opt]) for j in range(horizon)]
    avg_grd = [np.mean([ep["trajectory"][j][3] for ep in res_grd]) for j in range(horizon)]
    avg_rnd = [np.mean([ep["trajectory"][j][3] for ep in res_rnd]) for j in range(horizon)]

    print("Cumulative optimal reward:", avg_opt[-1] / horizon)
    value = res_opt[0]["value"]
    tot_r = res_opt[0]["total_reward"]
    dsc_r = res_opt[0]["discounted_return"]
    print("Value:", value)
    print("Tot_r:", tot_r)
    print("Dsc_r:", dsc_r)
    print("Tot_r/hor:", tot_r/horizon)
    print("Cumulative greedy reward:", avg_grd[-1] / horizon)
    print("Cumulative random reward:", avg_rnd[-1] / horizon)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(avg_opt, label="Optimal Policy", color="blue", linewidth=2)
    plt.plot(avg_grd, label="Greedy Policy", color="green", linestyle="--")
    # plt.plot(avg_rnd, label="Random Policy", color="red", linestyle=":")
    plt.fill_between(range(horizon), avg_opt, avg_grd, color="blue", alpha=0.1)
    plt.title("Cumulative Throughput (Long-term)")
    plt.xlabel("Steps in Episode")
    plt.ylabel("Total Reward Accumulated")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    sns.boxplot(data=[avg_opt, avg_grd, avg_rnd], palette=["blue", "green", "red"])
    plt.xticks([0, 1, 2], ["Optimal", "Greedy", "Random"])
    plt.title("Reward Variance per Episode")
    plt.ylabel("Total Episode Reward")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    comm_mdp, _ = build_comm_mdp(jammer="sweeper")

    policy, V = comm_mdp.solve_policy_iteration()
    greedy = comm_mdp.solve_greedy()

    run_comparison(comm_mdp, policy, greedy)
