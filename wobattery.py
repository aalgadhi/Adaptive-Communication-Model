import random
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from mdp_solver import MDP


def multi_drones_reward():
    pass

def build_comm_mdp(jammer="random"):
    # 1. Action Space
    channels = [1, 2, 3]
    mcs_levels = [1, 2, 3]
    power_levels = [1.0, 10.0]
    actions = [(f, m, p) for f in channels for m in mcs_levels for p in power_levels]

    jamming_vectors = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
    snr_vectors = [tuple(int(x) for x in bin(i)[2:].zfill(3)) for i in range(8)]
    states = [(J, C, mcs, f) for J in jamming_vectors for C in snr_vectors for mcs in mcs_levels for f in channels]

    T, R = {}, {}
    random.seed(42)

    for s in states:
        T[s], R[s] = {}, {}
        curr_J, curr_C, prev_mcs, prev_f = s
    
        jammer_next_dist = [(1/len(jamming_vectors), Jn) for Jn in jamming_vectors]  # uniform
        snr_dist = [(0.7, curr_C), (0.3, None)]  # None means "resample"

        for a in actions:
            target_f, target_mcs, target_p = a
            outcomes = []
            
            if jammer == "random":
                for pJ, next_J in jammer_next_dist:
                    # steady SNR branch
                    s_steady = (next_J, curr_C, target_mcs, target_f)
                    outcomes.append((pJ * 0.33, s_steady))

                    # change SNR branch: spread over all snr_vectors
                    pC = 0.67 / len(snr_vectors)
                    for next_C in snr_vectors:
                        s_change = (next_J, next_C, target_mcs, target_f)
                        outcomes.append((pJ * pC, s_change))
            
            elif jammer == "sweeper":
                next_jammer_idx = (jamming_vectors.index(curr_J) + 1) % len(jamming_vectors)
                next_J = jamming_vectors[next_jammer_idx]
                
                s_steady = (next_J, curr_C, target_mcs, target_f)
                s_change = (next_J, random.choice(snr_vectors), target_mcs, target_f)
                
                outcomes = [(0.33, s_steady), (0.67, s_change)]

            T[s][a] = outcomes
            R[s][a] = {}

            # 4. Reward Logic: Rate - Costs
            for prob, next_s in outcomes:
                nj, nc, _, _ = next_s
                if nj[target_f - 1] == 1:
                    reward = -20
                else:
                    rate = 30.0 if nc[target_f - 1] == 1 else 10.0
                    c_mcs = 5.0 if target_mcs != prev_mcs else 0.0
                    c_f = 10.0 if target_f != prev_f else 0.0
                    reward = rate - c_mcs - c_f

                R[s][a][next_s] = reward

    return MDP(states, actions, T, R), states

def plot_strategic_analysis(policy, states):
    data = []
    for s, a in policy.items():
        data.append({
            'Ch1_Jammed': s[0][0],
            'Ch1_SNR': s[1][0],
            'Action_Channel': a[0],
            'Action_MCS': a[1],
            'Chosen_Ch_SNR': s[1][a[0]-1]
        })
    df = pd.DataFrame(data)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Monotonicity: SNR vs MCS
    sns.barplot(x='Chosen_Ch_SNR', y='Action_MCS', data=df, ax=axes[0], palette='viridis')
    axes[0].set_title('Monotonicity: SNR vs MCS')

    # 2. SNR vs Frequency Selection
    sns.countplot(x='Ch1_SNR', hue='Action_Channel', data=df, ax=axes[1])
    axes[1].set_title('Frequency Selection vs Ch1 SNR')

    # 3. Jamming Awareness
    sns.countplot(x='Ch1_Jammed', hue='Action_Channel', data=df, ax=axes[2])
    axes[2].set_title('Frequency Selection vs Ch1 Jamming')

    plt.tight_layout()
    plt.savefig('strategic_analysis.png')
    plt.show()

def run_comparison(mdp, policy):
    start_state = random.choice(mdp.states)
    horizon = 50
    num_episodes = 100
    res_opt = mdp.simulate_monte_carlo(start_state, num_episodes=num_episodes, horizon=horizon, policy=policy)
    avg_opt = np.mean([r['total_reward'] for r in res_opt])
    avg_opt_reward_by_episode = [np.mean(list(ep['trajectory'][j][3] for ep in res_opt)) for j in range(horizon)]



    res_rnd = mdp.simulate_monte_carlo(start_state, num_episodes=num_episodes, horizon=horizon, policy=None)
    avg_rnd = np.mean([r['total_reward'] for r in res_rnd])
    avg_rnd_reward_by_episode = [np.mean(list(ep['trajectory'][j][3] for ep in res_rnd)) for j in range(horizon)]

    fig, ax = plt.subplots()
    
    x = list(range(horizon))

    plt.plot(x, avg_opt_reward_by_episode, 'r--', label="Optimal Policy") 
    plt.plot(x, avg_rnd_reward_by_episode, 'b--', label="Random Policy")

    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Average Reward by Episode vs Episode")
    plt.legend()
    plt.show()

    # plt.figure(figsize=(8, 5))
    # plt.bar(['Optimal Policy', 'Random Policy'], [avg_opt, avg_rnd], color=['#3498db', '#95a5a6'])
    # plt.title('Throughput Comparison (Avg Total Reward)')
    # plt.ylabel('Throughput Value')
    # plt.savefig('throughput_comparison.png')
    # plt.show()

    print(f"Optimal Policy Avg Reward: {avg_opt:.2f}")
    print(f"Random Policy Avg Reward: {avg_rnd:.2f}")

if __name__ == "__main__":
    comm_mdp, all_states = build_comm_mdp(jammer="random")
    policy, _ = comm_mdp.solve_policy_iteration()
    greedy, _ = comm_mdp.solve_greedy()
    # plot_strategic_analysis(policy, all_states)
    run_comparison(comm_mdp, policy)
