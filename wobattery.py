import random
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from mdp_solver import MDP


def comm_mdp_reward(state, next_state, action):
    curr_J, curr_C_idx, prev_mcs, prev_f, curr_batt = state
    nj, next_C_idx, _, _, next_batt = next_state
    target_f, target_mcs, target_p = action

    # Physical Constants
    W = 20e6
    N0_density = 4e-21
    N0 = N0_density * W
    P_jammer = 0.08
    G_jammer = 1e-4
    beta_th = 5.0

    gain_mapping = {i: 10**((i*3 - 20)/10) for i in range(8)}
    g_k = gain_mapping[next_C_idx]

    # Jammer interference if jammer is on chosen channel
    interference = P_jammer * G_jammer if nj[target_f - 1] == 1 else 0.0
    sinr = (target_p * g_k) / (interference + N0)

    # Shannon Throughput capped by MCS
    mcs_eff = {1: 1.5, 2: 3.0, 3: 5.0}
    theoretical_rate = W * math.log2(1 + sinr)
    throughput = min(theoretical_rate, mcs_eff[target_mcs] * W)

    # Costs
    alpha = 2.0
    energy_cost = alpha * target_p
    c_f = 10.0 if target_f != prev_f else 0.0
    c_mcs = 5.0 if target_mcs != prev_mcs else 0.0

    reward = (throughput / 1e6) - energy_cost - c_f - c_mcs

    # Outage penalty
    if sinr < beta_th:
        reward -= 50.0

    if curr_batt == 0:
        reward = 0.0                 # "dead battery" catastrophe
    elif curr_batt == 1:
        reward -= 20.0                  # "low battery" discomfort

    if next_batt == 0:
        reward -= 50.0
    
    reward = reward if curr_batt else 0
    return reward


def build_comm_mdp(jammer="random"):
    # 1) Action Space
    channels = [1, 2, 3]
    mcs_levels = [1, 2, 3]
    power_levels = [0.00016, 0.001, 0.01]  # 0.16mW, 1mW, 10mW
    actions = [(f, m, p) for f in channels for m in mcs_levels for p in power_levels]

    # 2) Physical State Space
    jamming_vectors = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]

    # FSMC channel states (0..7)
    snr_states = list(range(8))

    # Battery levels
    battery_levels = list(range(20, -1, -1))

    # State = (JammerVector, ChannelIdx, PrevMCS, PrevFreq, Battery)
    states = [
        (J, C_idx, prev_mcs, prev_f, b)
        for J in jamming_vectors
        for C_idx in snr_states
        for prev_mcs in mcs_levels
        for prev_f in channels
        for b in battery_levels
    ]

    T, R = {}, {}
    random.seed(42)

    def batt_drop(p):
        """Battery depletion per step based on TX power."""
        if p >= 0.01:       # high power
            return 3
        elif p >= 0.001:    # mid power
            return 1
        else:               # low power
            return 0

    for s in states:
        T[s], R[s] = {}, {}
        curr_J, curr_C_idx, prev_mcs, prev_f, curr_batt = s

        for a in actions:
            target_f, target_mcs, target_p = a

            # --- Jammer transition depends on action (reactive jammer) ---
            if jammer == "sweeper":
                next_jammer_idx = (jamming_vectors.index(curr_J) + 1) % len(jamming_vectors)
                next_jammer_vectors = [jamming_vectors[next_jammer_idx]]
                jammer_probs = [1.0]

            elif jammer == "reactive":
                # Higher power => easier detection => jammer follows more strongly
                p_follow = 0.6 + 0.3 * (target_p / max(power_levels))
                p_follow = min(0.9, max(0.6, p_follow))

                jammer_probs_full = [ (1.0 - p_follow) / 2 ] * 3
                jammer_probs_full[target_f - 1] = p_follow

                next_jammer_vectors = jamming_vectors
                jammer_probs = jammer_probs_full

            else:
                # random jammer
                next_jammer_vectors = jamming_vectors
                jammer_probs = [1/3, 1/3, 1/3]

            # --- FSMC Channel Transition (action-independent) ---
            if curr_C_idx == 0:
                possible_C = [(0.8, 0), (0.2, 1)]
            elif curr_C_idx == 7:
                possible_C = [(0.2, 6), (0.8, 7)]
            else:
                possible_C = [(0.15, curr_C_idx - 1), (0.7, curr_C_idx), (0.15, curr_C_idx + 1)]

            # --- Battery transition depends on power ---
            drop = batt_drop(target_p)
            next_batt = max(0, curr_batt - drop)

            # Combine Jammer + Channel transitions into next states
            outcomes = []
            for pJ, next_J in zip(jammer_probs, next_jammer_vectors):
                for pC, next_C_idx in possible_C:
                    prob = pJ * pC
                    # Next state stores chosen (mcs,f) as the new "prev" and updated battery
                    next_s = (next_J, next_C_idx, target_mcs, target_f, next_batt)
                    outcomes.append((prob, next_s))

            T[s][a] = outcomes
            R[s][a] = {}

            for prob, next_s in outcomes:
                reward = comm_mdp_reward(s, next_s, a)
                R[s][a][next_s] = reward

    return MDP(states, actions, T, R), states


def plot_strategic_analysis(policy):
    """
    Visualizes the policy as a heatmap. 
    Shows which channel the agent chooses based on Jammer location vs SNR state.
    """
    # Define our state space dimensions for the grid
    jammer_vectors = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
    snr_vectors = [tuple(int(x) for x in bin(i)[2:].zfill(3)) for i in range(8)]
    
    # We fix MCS and Prev_Freq to get a 2D slice of the policy
    fixed_mcs = 1
    fixed_prev_f = 1
    
    # Create a 2D array to store the chosen frequency (Action[0])
    policy_grid = np.zeros((len(jammer_vectors), len(snr_vectors)))

    for i, J in enumerate(jammer_vectors):
        for j, C in enumerate(snr_vectors):
            state = (J, C, fixed_mcs, fixed_prev_f)
            if state in policy:
                # Store the chosen frequency (index 0 of the action tuple)
                policy_grid[i, j] = policy[state][0]

    plt.figure(figsize=(10, 5))
    sns.heatmap(policy_grid, annot=True, cmap="YlGnBu", cbar_kws={'label': 'Selected Channel'})
    
    plt.title(f"Policy Map: Selected Channel\n(Fixed MCS: {fixed_mcs}, Prev Freq: {fixed_prev_f})")
    plt.xlabel("SNR State Index (0-7)")
    plt.ylabel("Jammer Active Channel (1-3)")
    plt.yticks(ticks=[0.5, 1.5, 2.5], labels=['CH 1', 'CH 2', 'CH 3'])
    plt.show()



def plot_3d_action_manifold(policy):
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Fixed: Jammer on Ch 1, Prev Freq 2, Prev MCS 1
    fixed_J = (1, 0, 0)
    fixed_prev_f = 2
    fixed_prev_mcs = 1

    snr_indices = list(range(8))
    battery_levels = list(range(21))

    X, Y, Z, C = [], [], [], []

    for b in battery_levels:
        for s_idx in snr_indices:
            state = (fixed_J, s_idx, fixed_prev_mcs, fixed_prev_f, b)
            if state in policy:
                action = policy[state]
                X.append(b)           # Battery Level
                Y.append(s_idx)       # SNR Index
                Z.append(action[2])   # Target Power (Action)
                C.append(action[1])   # MCS level (Action) - Color dimension

    sc = ax.scatter(X, Y, Z, c=C, cmap='plasma', s=100, alpha=0.8, edgecolors='w')
    ax.set_xlabel('Current Battery Level')
    ax.set_ylabel('Channel Quality (SNR Index)')
    ax.set_zlabel('Target Transmit Power (W)')
    plt.title("3D Policy Manifold: Power & MCS adaptation based on Energy/SNR")
    cbar = plt.colorbar(sc, ax=ax, shrink=0.5)
    cbar.set_label('Chosen MCS Level')
    plt.show()


def plot_monotonicity_variable_analysis(policy):
    snr_indices = list(range(8))
    # Aggregate actions across all battery/jammer states to see general trends
    avg_mcs = []
    avg_p = []
    
    for s_idx in snr_indices:
        actions_at_snr = [a for s, a in policy.items() if s[1] == s_idx and s[4] > 5] # Only for healthy battery
        avg_mcs.append(np.mean([a[1] for a in actions_at_snr]))
        avg_p.append(np.mean([a[2] for a in actions_at_snr]))

    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.set_xlabel('SNR Index (Worse -> Better Channel)')
    ax1.set_ylabel('Mean MCS Choice', color='tab:blue')
    ax1.plot(snr_indices, avg_mcs, color='tab:blue', marker='o', lw=3, label="MCS Trend")
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Mean Power Choice (W)', color='tab:red')
    ax2.plot(snr_indices, avg_p, color='tab:red', marker='x', linestyle='--', lw=2, label="Power Trend")
    ax2.tick_params(axis='y', labelcolor='tab:red')

    plt.title("Policy Monotonicity: Action Response to Channel Quality")
    fig.tight_layout()
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_jammer_avoidance_heatmap(policy):
    # X: Jammer Channel (1-3), Y: Drone Target Channel (1-3)
    # Intensity: How often this pair occurs in the policy
    heatmap_data = np.zeros((3, 3))
    
    for s, a in policy.items():
        jammer_ch = np.argmax(s[0]) # 0, 1, or 2
        target_ch = a[0] - 1        # 0, 1, or 2
        heatmap_data[jammer_ch, target_ch] += 1

    # Normalize by row
    row_sums = heatmap_data.sum(axis=1)
    heatmap_data = heatmap_data / row_sums[:, np.newaxis]

    plt.figure(figsize=(8, 6))
    sns.heatmap(heatmap_data, annot=True, cmap="Reds", 
                xticklabels=['Target Ch 1', 'Target Ch 2', 'Target Ch 3'],
                yticklabels=['Jammer Ch 1', 'Jammer Ch 2', 'Jammer Ch 3'])
    plt.title("Jammer Avoidance Probability Heatmap\n(Optimal Policy Distribution)")
    plt.show()


def run_comparison(mdp, policy, greedy):
    start_state = random.choice(mdp.states)
    horizon = 100
    num_episodes = 10

    # Simulate all three
    res_opt = mdp.simulate_monte_carlo(start_state, num_episodes=num_episodes, horizon=horizon, policy=policy)
    res_grd = mdp.simulate_monte_carlo(start_state, num_episodes=num_episodes, horizon=horizon, policy=greedy)
    res_rnd = mdp.simulate_monte_carlo(start_state, num_episodes=num_episodes, horizon=horizon, policy=None)

    # Calculate Average Up-to-Step Reward
    avg_opt = [np.mean([ep['trajectory'][j][3] for ep in res_opt]) for j in range(horizon)]
    avg_grd = [np.mean([ep['trajectory'][j][3] for ep in res_grd]) for j in range(horizon)]
    avg_rnd = [np.mean([ep['trajectory'][j][3] for ep in res_rnd]) for j in range(horizon)]
    
    print("Cumulative optimal reward:", avg_opt[-1])
    print("Cumulative greedy reward:", avg_grd[-1])
    print("Cumulative random reward:", avg_rnd[-1])


    # Plot 1: Cumulative Reward Comparison
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(avg_opt, label="Optimal Policy", color='blue', linewidth=2)
    plt.plot(avg_grd, label="Greedy Policy", color='green', linestyle='--')
    plt.plot(avg_rnd, label="Random Policy", color='red', linestyle=':')
    plt.fill_between(range(horizon), avg_opt, avg_grd, color='blue', alpha=0.1)
    plt.title("Cumulative Throughput (Long-term)")
    plt.xlabel("Steps in Episode")
    plt.ylabel("Total Reward Accumulated")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 2: Reward Distribution (Boxplot)
    plt.subplot(1, 2, 2)
    data_to_plot = [
        [np.mean([ep['trajectory'][j][2] for ep in res_opt]) for j in range(horizon)],
        [np.mean([ep['trajectory'][j][2] for ep in res_grd]) for j in range(horizon)],
        [np.mean([ep['trajectory'][j][2] for ep in res_rnd]) for j in range(horizon)]
    ]
    sns.boxplot(data=data_to_plot, palette=["blue", "green", "red"])
    plt.xticks([0, 1, 2], ['Optimal', 'Greedy', 'Random'])
    plt.title("Reward Variance per Episode")
    plt.ylabel("Total Episode Reward")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    comm_mdp, all_states = build_comm_mdp(jammer="random")
    policy, V = comm_mdp.solve_policy_iteration()
    greedy = comm_mdp.solve_greedy()

    plot_strategic_analysis(policy)
    plot_3d_action_manifold(policy)
    plot_monotonicity_variable_analysis(policy)
    plot_jammer_avoidance_heatmap(policy)
    run_comparison(comm_mdp, policy, greedy)