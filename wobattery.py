import math
import random

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from mdp_solver import MDP

# Random seed
SEED = 42

# -----------------------------
# Li-inspired anti-jamming setup
# -----------------------------
CHANNELS = [1, 2, 3]
JAMMING_VECTORS = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]

# Li's comparison section uses three low transmit powers.
POWER_LEVELS_DBM = [-8.5, -7.5]
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

    total_cost = math.log(1e4 * delay_cost) + mcs_switch_cost + freq_switch_cost + power_cost + 5 * (next_jammer_state[target_f-1] == 1)

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


def plot_strategic_analysis(policy):
    """Heatmap: selected channel vs jammer position and chosen channel's own quality."""
    fixed_prev_mcs = 1
    fixed_prev_f = 1
    fixed_other_channels = (3, 3)

    policy_grid = np.zeros((len(JAMMING_VECTORS), len(SNR_STATES)))

    for i, J in enumerate(JAMMING_VECTORS):
        for j, c1 in enumerate(SNR_STATES):
            state = (J, (c1, fixed_other_channels[0], fixed_other_channels[1]), fixed_prev_mcs, fixed_prev_f)
            if state in policy:
                policy_grid[i, j] = policy[state][0]

    plt.figure(figsize=(10, 5))
    sns.heatmap(policy_grid, annot=True, cmap="YlGnBu", cbar_kws={"label": "Selected Channel"})
    plt.title(
        "Policy Map: Selected Channel\n"
        f"(C2={fixed_other_channels[0]}, C3={fixed_other_channels[1]}, Prev MCS={fixed_prev_mcs}, Prev Freq={fixed_prev_f})"
    )
    plt.xlabel("Channel 1 Quality State (0-7)")
    plt.ylabel("Jammer Active Channel")
    plt.yticks(ticks=[0.5, 1.5, 2.5], labels=["CH 1", "CH 2", "CH 3"])
    plt.tight_layout()
    plt.show()


def plot_3d_action_manifold(policy):
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    fixed_J = (1, 0, 0)
    fixed_prev_f = 2
    fixed_prev_mcs = 1
    fixed_c3 = 3

    x_vals, y_vals, z_vals, colors = [], [], [], []

    for c1 in SNR_STATES:
        for c2 in SNR_STATES:
            state = (fixed_J, (c1, c2, fixed_c3), fixed_prev_mcs, fixed_prev_f)
            if state in policy:
                action = policy[state]
                x_vals.append(c1)
                y_vals.append(c2)
                z_vals.append(action[2])
                colors.append(action[1])

    sc = ax.scatter(x_vals, y_vals, z_vals, c=colors, cmap="plasma", s=100, alpha=0.8, edgecolors="w")
    ax.set_xlabel("Channel 1 Quality")
    ax.set_ylabel("Channel 2 Quality")
    ax.set_zlabel("Target Transmit Power (W)")
    plt.title(f"Power/MCS Adaptation with Independent Channel States (C3 fixed at {fixed_c3})")
    cbar = plt.colorbar(sc, ax=ax, shrink=0.5)
    cbar.set_label("Chosen MCS Level")
    plt.tight_layout()
    plt.show()


def plot_monotonicity_variable_analysis(policy):
    avg_mcs = []
    avg_p = []

    for chosen_quality in SNR_STATES:
        matching_actions = []
        for s, a in policy.items():
            _, c_tuple, _, _ = s
            target_channel = a[0] - 1
            if c_tuple[target_channel] == chosen_quality:
                matching_actions.append(a)

        avg_mcs.append(np.mean([a[1] for a in matching_actions]) if matching_actions else np.nan)
        avg_p.append(np.mean([a[2] for a in matching_actions]) if matching_actions else np.nan)

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.set_xlabel("Chosen Channel Quality State")
    ax1.set_ylabel("Mean MCS Choice", color="tab:blue")
    ax1.plot(SNR_STATES, avg_mcs, color="tab:blue", marker="o", lw=3)
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    ax2.set_ylabel("Mean Power Choice (W)", color="tab:red")
    ax2.plot(SNR_STATES, avg_p, color="tab:red", marker="x", linestyle="--", lw=2)
    ax2.tick_params(axis="y", labelcolor="tab:red")

    plt.title("Policy Monotonicity: Action Response to Selected Channel Quality")
    fig.tight_layout()
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_jammer_avoidance_heatmap(policy):
    heatmap_data = np.zeros((3, 3))

    for s, a in policy.items():
        jammer_ch = np.argmax(s[0])
        target_ch = a[0] - 1
        heatmap_data[jammer_ch, target_ch] += 1

    row_sums = heatmap_data.sum(axis=1, keepdims=True)
    heatmap_data = np.divide(heatmap_data, row_sums, out=np.zeros_like(heatmap_data), where=row_sums != 0)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        heatmap_data,
        annot=True,
        cmap="Reds",
        xticklabels=["Target Ch 1", "Target Ch 2", "Target Ch 3"],
        yticklabels=["Jammer Ch 1", "Jammer Ch 2", "Jammer Ch 3"],
    )
    plt.title("Jammer Avoidance Probability Heatmap\n(Optimal Policy Distribution)")
    plt.tight_layout()
    plt.show()


def plot_delay_curves_like_paper(save_path=None):
    """Paper-style delay curves versus effective SINR for each MCS and power level."""
    fig, ax = plt.subplots(figsize=(9, 6))
    marker_cycle = ["o", "^", "s"]

    for p_idx, power in enumerate(POWER_LEVELS):
        sinr_points = []
        delay_curves = {m: [] for m in MCS_LEVELS}
        for state_idx in SNR_STATES:
            jammer = (0, 0, 1)
            channel_tuple = (state_idx, 4, 4)
            action_base = (1, 1, power)
            metrics_base = _selected_link_metrics(jammer, channel_tuple, action_base)
            sinr_points.append(metrics_base["sinr_db"])
            for m in MCS_LEVELS:
                metrics = _selected_link_metrics(jammer, channel_tuple, (1, m, power))
                expected_good_rate = max(metrics["goodput"], 1e-9)
                delay_curves[m].append((1.0 / expected_good_rate) * 1e4)

        for m in MCS_LEVELS:
            ax.semilogy(
                sinr_points,
                delay_curves[m],
                marker=marker_cycle[p_idx],
                linewidth=1.5,
                markersize=4,
                label=f"{MCS_LABELS[m]}, {POWER_LABELS[power]}",
            )

    ax.set_title("Delay Cost vs. Effective SINR")
    ax.set_xlabel("Effective SINR (dB)")
    ax.set_ylabel("Delay Cost × 1e4")
    ax.grid(True, which="both", linestyle=":", alpha=0.5)
    ax.legend(fontsize=8, ncol=3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()


def plot_optimal_mcs_vs_sinr_like_paper(policy, save_path=None):
    """Paper-style threshold plot of optimal MCS versus SINR for several previous MCS values."""
    fig, ax = plt.subplots(figsize=(9, 6))
    fixed_jammer = (0, 1, 0)
    fixed_other = (4, 4)
    fixed_prev_f = 1
    markers = ["o", "^", "s"]

    for idx, prev_mcs in enumerate(MCS_LEVELS):
        sinr_values = []
        chosen_mcs = []
        for c1 in SNR_STATES:
            state = (fixed_jammer, (c1, fixed_other[0], fixed_other[1]), prev_mcs, fixed_prev_f)
            action = policy[state]
            metrics = _selected_link_metrics(fixed_jammer, (c1, fixed_other[0], fixed_other[1]), (1, action[1], action[2]))
            sinr_values.append(metrics["sinr_db"])
            chosen_mcs.append(action[1])
        ax.step(sinr_values, chosen_mcs, where="post", marker=markers[idx], linewidth=1.8, label=f"Prev MCS={prev_mcs}")

    ax.set_title("Optimal MCS vs. Effective SINR")
    ax.set_xlabel("Effective SINR (dB)")
    ax.set_ylabel("Optimal MCS")
    ax.set_yticks(MCS_LEVELS)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()


def extract_policy_statistics(policy, switching_cost_values):
    """Evaluate throughput and switching rate under different MCS switching penalties."""
    base_state = representative_state(prev_mcs=1, prev_f=1, jammer=(0, 1, 0), channel_tuple=(2, 4, 6))
    stats = {"switch_cost": [], "throughput": [], "switch_rate": []}

    original_coeff = globals().get("MCS_SWITCH_COST_COEFF", 5.0)
    for coeff in switching_cost_values:
        globals()["MCS_SWITCH_COST_COEFF"] = coeff
        mdp, _ = build_comm_mdp(jammer="random")
        local_policy, _ = mdp.solve_policy_iteration()

        start_state = base_state
        episodes = mdp.simulate_monte_carlo(start_state, num_episodes=6, horizon=50, policy=local_policy)
        throughputs = []
        switch_counts = []
        for ep in episodes:
            traj = ep["trajectory"]
            state = start_state
            prev_action = None
            for step in traj:
                action = local_policy[state]
                link = _selected_link_metrics(state[0], state[1], action)
                throughputs.append(link["goodput"] / 1e6)
                if prev_action is not None and action[1] != prev_action[1]:
                    switch_counts.append(1)
                else:
                    switch_counts.append(0)
                prev_action = action
                state = step[1]

        stats["switch_cost"].append(coeff)
        stats["throughput"].append(float(np.mean(throughputs)))
        stats["switch_rate"].append(float(np.mean(switch_counts) * 100.0))

    globals()["MCS_SWITCH_COST_COEFF"] = original_coeff
    return stats


def plot_switching_tradeoff_like_paper(save_prefix=None):
    switch_cost_values = [0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0]
    stats = extract_policy_statistics(None, switch_cost_values)

    fig, axes = plt.subplots(2, 1, figsize=(8, 9), sharex=True)

    axes[0].plot(stats["switch_cost"], stats["throughput"], marker="s", linewidth=2)
    axes[0].set_title("Throughput vs. MCS Switching Cost")
    axes[0].set_ylabel("Mean Goodput (Mbit/s)")
    axes[0].grid(True, linestyle=":", alpha=0.5)

    axes[1].plot(stats["switch_cost"], stats["switch_rate"], marker="o", linewidth=2)
    axes[1].set_title("MCS Switching Rate vs. MCS Switching Cost")
    axes[1].set_xlabel("Switching Cost Coefficient")
    axes[1].set_ylabel("MCS Switch Rate (%)")
    axes[1].grid(True, linestyle=":", alpha=0.5)

    fig.tight_layout()
    if save_prefix:
        fig.savefig(f"{save_prefix}_tradeoff.png", dpi=200, bbox_inches="tight")
    plt.show()


def plot_paper_style_summary(policy, save_path=None):
    """Create a 2x2 summary figure similar to the paper layout."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: delay curves
    marker_cycle = ["o", "^", "s"]
    for p_idx, power in enumerate(POWER_LEVELS):
        sinr_points = []
        delay_curves = {m: [] for m in MCS_LEVELS}
        for state_idx in SNR_STATES:
            jammer = (0, 0, 1)
            channel_tuple = (state_idx, 4, 4)
            metrics_base = _selected_link_metrics(jammer, channel_tuple, (1, 1, power))
            sinr_points.append(metrics_base["sinr_db"])
            for m in MCS_LEVELS:
                metrics = _selected_link_metrics(jammer, channel_tuple, (1, m, power))
                delay_curves[m].append((1.0 / max(metrics["goodput"], 1e-9)) * 1e4)
        for m in MCS_LEVELS:
            axes[0, 0].semilogy(
                sinr_points,
                delay_curves[m],
                marker=marker_cycle[p_idx],
                linewidth=1.2,
                markersize=3,
                label=f"{MCS_LABELS[m]}, {POWER_LABELS[power]}",
            )
    axes[0, 0].set_title("Delay Costs vs. Effective SINR")
    axes[0, 0].set_xlabel("Effective SINR (dB)")
    axes[0, 0].set_ylabel("Delay Cost × 1e4")
    axes[0, 0].grid(True, which="both", linestyle=":", alpha=0.5)
    axes[0, 0].legend(fontsize=7, ncol=2)

    # Panel 2: throughput vs switching cost
    stats = extract_policy_statistics(policy, [0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0])
    axes[0, 1].plot(stats["switch_cost"], stats["throughput"], marker="s", linewidth=2)
    axes[0, 1].set_title("Throughput vs. Switching Cost")
    axes[0, 1].set_xlabel("MCS Switching Cost Coefficient")
    axes[0, 1].set_ylabel("Mean Goodput (Mbit/s)")
    axes[0, 1].grid(True, linestyle=":", alpha=0.5)

    # Panel 3: optimal MCS vs SINR
    fixed_jammer = (0, 1, 0)
    fixed_other = (4, 4)
    fixed_prev_f = 1
    markers = ["o", "^", "s"]
    for idx, prev_mcs in enumerate(MCS_LEVELS):
        sinr_values = []
        chosen_mcs = []
        for c1 in SNR_STATES:
            state = (fixed_jammer, (c1, fixed_other[0], fixed_other[1]), prev_mcs, fixed_prev_f)
            action = policy[state]
            metrics = _selected_link_metrics(fixed_jammer, (c1, fixed_other[0], fixed_other[1]), (1, action[1], action[2]))
            sinr_values.append(metrics["sinr_db"])
            chosen_mcs.append(action[1])
        axes[1, 0].step(sinr_values, chosen_mcs, where="post", marker=markers[idx], linewidth=1.8, label=f"Prev MCS={prev_mcs}")
    axes[1, 0].set_title("Optimal MCS vs. Effective SINR")
    axes[1, 0].set_xlabel("Effective SINR (dB)")
    axes[1, 0].set_ylabel("Optimal MCS")
    axes[1, 0].set_yticks(MCS_LEVELS)
    axes[1, 0].grid(True, linestyle=":", alpha=0.5)
    axes[1, 0].legend(fontsize=8)

    # Panel 4: switching rate vs switching cost
    axes[1, 1].plot(stats["switch_cost"], stats["switch_rate"], marker="o", linewidth=2)
    axes[1, 1].set_title("Switching Rate vs. Switching Cost")
    axes[1, 1].set_xlabel("MCS Switching Cost Coefficient")
    axes[1, 1].set_ylabel("MCS Switch Rate (%)")
    axes[1, 1].grid(True, linestyle=":", alpha=0.5)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()


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

    print("Cumulative optimal reward:", avg_opt[-1] / len(avg_opt))
    print("Cumulative greedy reward:", avg_grd[-1] / len(avg_grd))
    print("Cumulative random reward:", avg_rnd[-1] / len(avg_rnd))

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
    comm_mdp, _ = build_comm_mdp(jammer="reactive")

    policy, V = comm_mdp.solve_policy_iteration()
    greedy = comm_mdp.solve_greedy()

    # plot_paper_style_summary(policy, save_path="wobattery_paper_style_summary.png")
    # plot_delay_curves_like_paper(save_path="wobattery_delay_curves.png")
    # plot_optimal_mcs_vs_sinr_like_paper(policy, save_path="wobattery_optimal_mcs.png")
    # plot_switching_tradeoff_like_paper(save_prefix="")

    plot_strategic_analysis(policy)
    plot_3d_action_manifold(policy)
    plot_monotonicity_variable_analysis(policy)
    plot_jammer_avoidance_heatmap(policy)
    run_comparison(comm_mdp, policy, greedy)
