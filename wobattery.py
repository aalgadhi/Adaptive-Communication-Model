import random
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from mdp_solver import MDP


def multi_drones_reward():
    pass

def comm_mdp_reward(state, next_state, action):
    curr_J, curr_C, prev_mcs, prev_f = state
    nj, nc, _, _ = next_state
    target_f, target_mcs, target_p = action

    if nj[target_f - 1] == 1:
        reward = -20
    else:
        rate = 30.0 if nc[target_f - 1] == 1 else 10.0
        c_mcs = 5.0 if target_mcs != prev_mcs else 0.0
        c_f = 10.0 if target_f != prev_f else 0.0
        reward = rate - c_mcs - c_f
    
    return reward



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
                reward = comm_mdp_reward(s, next_s, a)
                R[s][a][next_s] = reward

    return MDP(states, actions, T, R), states

def plot_strategic_analysis(policy, states):
    data = []
    channel1_snr = []
    freq_chosen = []
    for s, a in policy.items():
        if s[0][0] == 0 and not s[1][1] and not s[1][2] and s[2] == 1:
            channel1_snr.append(s[1][0])
            freq_chosen.append(a[0])


    data = {
        "Channel 1 SNR": channel1_snr,
        "Chosen Freq": freq_chosen
    }
    print(data)

    sns.lineplot(data, x="Channel 1 SNR", y="Chosen Freq")
    
   
def run_comparison(mdp, policy, greedy):
    start_state = random.choice(mdp.states)
    horizon = 50
    num_episodes = 100

    res_opt = mdp.simulate_monte_carlo(start_state, num_episodes=num_episodes, horizon=horizon, policy=policy)
    avg_opt = np.mean([r['total_reward'] for r in res_opt])
    avg_opt_reward_by_episode = [np.mean(list(ep['trajectory'][j][3] for ep in res_opt)) for j in range(horizon)]


    res_grd = mdp.simulate_monte_carlo(start_state, num_episodes=num_episodes, horizon=horizon, policy=greedy)
    avg_grd = np.mean([r['total_reward'] for r in res_grd])
    avg_grd_reward_by_episode = [np.mean(list(ep['trajectory'][j][3] for ep in res_grd)) for j in range(horizon)]


    res_rnd = mdp.simulate_monte_carlo(start_state, num_episodes=num_episodes, horizon=horizon, policy=None)
    avg_rnd = np.mean([r['total_reward'] for r in res_rnd])
    avg_rnd_reward_by_episode = [np.mean(list(ep['trajectory'][j][3] for ep in res_rnd)) for j in range(horizon)]

    fig, ax = plt.subplots()
    
    x = list(range(horizon))

    plt.plot(x, avg_opt_reward_by_episode, 'r--', label="Optimal Policy") 
    plt.plot(x, avg_grd_reward_by_episode, 'g--', label="Greedy Policy")
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
    print(f"Greedy Policy Avg Reward: {avg_grd:.2f}")
    print(f"Random Policy Avg Reward: {avg_rnd:.2f}")

if __name__ == "__main__":
    comm_mdp, all_states = build_comm_mdp(jammer="sweeper")
    policy, _ = comm_mdp.solve_policy_iteration()
    greedy = comm_mdp.solve_greedy()
    plot_strategic_analysis(policy, all_states)
    run_comparison(comm_mdp, policy, greedy)
