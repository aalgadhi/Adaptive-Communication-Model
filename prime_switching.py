import itertools
import time
from mdp_solver import MDP

class PrimeSwitchingMDP(MDP):
    def __init__(self, freqs, mcs_levels, channel_states, freq_switch_cost, mcs_switch_cost, gamma=0.9):
        self.freqs = freqs                  
        self.mcs_levels = mcs_levels        
        self.channel_states = channel_states 
        
        self.freq_vector = tuple(self.freqs)
        self.k_freq = freq_switch_cost
        self.k_mcs = mcs_switch_cost
        
        # 1. State Space 
        print("Generating Channel Vectors...")
        channel_vectors = list(itertools.product(self.channel_states, repeat=len(self.freqs)))
        
        self.states_list = []
        for ch_vec in channel_vectors:
            for prev_m in self.mcs_levels:
                for prev_f in self.freqs:
                    s = (ch_vec, prev_m, self.freq_vector, prev_f)
                    self.states_list.append(s)
                    
        # 2. Action Space 
        self.actions_list = list(itertools.product(self.mcs_levels, self.freqs))
        
        # 3. Transitions & Rewards
        transitions = {}
        rewards = {}
        
        # 5-State Transition Matrix
        self.P_ch = {
            0: {0: 0.6, 1: 0.4, 2: 0.0, 3: 0.0, 4: 0.0},
            1: {0: 0.2, 1: 0.5, 2: 0.3, 3: 0.0, 4: 0.0},
            2: {0: 0.0, 1: 0.2, 2: 0.5, 3: 0.3, 4: 0.0},
            3: {0: 0.0, 1: 0.0, 2: 0.2, 3: 0.5, 4: 0.3},
            4: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.2, 4: 0.8} 
        }
        
        print(f"Building Model Transitions for {len(self.states_list)} states...")
        start_time = time.time()
        
        for s in self.states_list:
            curr_ch_vec, prev_mcs, _, prev_freq = s
            transitions[s] = {}
            rewards[s] = {}
            
            # OPTIMIZATION: Reachable states only
            reachable_per_freq = []
            for i in range(len(self.freqs)):
                curr_c = curr_ch_vec[i]
                possible_next = [n for n, p in self.P_ch[curr_c].items() if p > 0]
                reachable_per_freq.append(possible_next)
            
            reachable_vectors = list(itertools.product(*reachable_per_freq))

            for a in self.actions_list:
                act_mcs, act_freq = a
                
                # Immediate Cost Logic
                quality_at_chosen_freq = curr_ch_vec[act_freq]
                
                # Calculate Effective Rate (including FER penalty)
                rate = self.get_rate(act_mcs, quality_at_chosen_freq)
                
                if rate <= 1e-2:
                    delay_cost = 100.0 
                else:
                    delay_cost = 100.0 / rate
                
                c_freq = self.k_freq if act_freq != prev_freq else 0.0
                c_mcs = self.k_mcs if act_mcs != prev_mcs else 0.0
                
                total_cost = delay_cost + c_freq + c_mcs
                                                
                outcomes = []
                
                for next_ch_vec in reachable_vectors:
                    prob_vec = 1.0
                    for i in range(len(self.freqs)):
                        prob_vec *= self.P_ch[curr_ch_vec[i]][next_ch_vec[i]]
                    
                    next_state = (next_ch_vec, act_mcs, self.freq_vector, act_freq)
                    outcomes.append((prob_vec, next_state))
                    
                    if a not in rewards[s]:
                        rewards[s][a] = {}
                    rewards[s][a][next_state] = -total_cost

                transitions[s][a] = outcomes
        
        print(f"Model Built in {time.time() - start_time:.2f} seconds.")
        super().__init__(self.states_list, self.actions_list, transitions, rewards, gamma)

    def get_rate(self, mcs, channel_quality):
        raw_rate = mcs * 10 + 10
        required_state = mcs + 1 
        
        diff = channel_quality - required_state
        
        if diff >= 1:
            efficiency = 1.0

        elif diff == 0:
            efficiency = 0.7 
            
        else:
            if mcs == 0 and channel_quality >= 1:
                efficiency = 0.5
            else:
                efficiency = 0.0
                
        return raw_rate * efficiency

if __name__ == "__main__":
    frequencies = [0, 1, 2, 3] 
    mcs_options = [0, 1, 2, 3]
    ch_states = [0, 1, 2, 3, 4]   
    
    cost_freq_switch = 5.0
    cost_mcs_switch = 2.0
    
    ps_mdp = PrimeSwitchingMDP(frequencies, mcs_options, ch_states, 
                               cost_freq_switch, cost_mcs_switch, gamma=0.9)
    
    print(f"State Space Size: {len(ps_mdp.states_list)}")
    
    policy, value_func = ps_mdp.solve_policy_iteration()

    test_state = ((2, 4, 0, 0), 1, tuple(frequencies), 0)
    
    print(f"\nScenario: F0=Okay(2), F1=Excellent(4). Currently on F0.")
    action = policy[test_state]
    print(f"Optimal Action: Switch to Freq {action[1]}, MCS {action[0]}")
    
    sim = ps_mdp.simulate_monte_carlo(test_state, num_episodes=1, horizon=15, policy=policy)
    for step in sim[0]['trajectory']:
        s, a, r = step
        ch_vec_str = ", ".join(str(x) for x in s[0])
        print(f"State:[{ch_vec_str}] at channel F{s[3]} -> Act: MCS{a[0]}/F{a[1]} Cost:{-r}")

    print("Random Walk Policy")
    sim = ps_mdp.simulate_monte_carlo(test_state, num_episodes=1, horizon=15, policy=None)
    for step in sim[0]['trajectory']:
        s, a, r = step
        ch_vec_str = ", ".join(str(x) for x in s[0])
        print(f"State:[{ch_vec_str}] at channel F{s[3]} -> Act: MCS{a[0]}/F{a[1]} Cost:{-r}")