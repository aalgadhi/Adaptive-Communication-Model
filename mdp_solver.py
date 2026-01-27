import random
import time

class MDP:
    def __init__(self, states, actions, transitions, rewards, gamma=0.9, epsilon=1e-6):
        self.states = states
        self.actions = actions
        self.transitions = transitions
        self.rewards = rewards
        self.gamma = gamma
        self.epsilon = epsilon

        # Initialize value function V(s) to 0
        self.V = {s: 0.0 for s in self.states}
        
        # Initialize policy pi(s) arbitrarily (pick the first available action)
        self.policy = {s: self.actions[0] for s in self.states}

    def get_transition_prob(self, state, action, next_state):
        if state not in self.transitions or action not in self.transitions[state]:
            return 0.0
        
        # Search through the list of possible outcomes
        for prob, ns in self.transitions[state][action]:
            if ns == next_state:
                return prob
        return 0.0

    def get_reward(self, state, action, next_state):
        """Helper to get Reward R(state, action, next_state)."""
        try:
            return self.rewards[state][action][next_state]
        except KeyError:
            return 0.0

    def policy_evaluation(self):
        while True:
            delta = 0
            # Iterate over all states to update V(s)
            for s in self.states:
                v_old = self.V[s]
                a = self.policy[s]
                
                # Calculate expected value for taking action 'a' in state 's'
                # V(s) = sum_s' [ P(s'|s,pi(s)) * (R(s,pi(s),s') + gamma * V(s')) ]
                new_v = 0
                if s in self.transitions and a in self.transitions[s]:
                    for prob, next_s in self.transitions[s][a]:
                        r = self.get_reward(s, a, next_s)
                        new_v += prob * (r + self.gamma * self.V[next_s])
                
                self.V[s] = new_v
                delta = max(delta, abs(v_old - self.V[s]))
            
            if delta < self.epsilon:
                break

    def policy_improvement(self):
        policy_stable = True
        
        for s in self.states:
            old_action = self.policy[s]
            best_action = None
            max_q_value = float('-inf')
            
            possible_actions = self.actions
            if s in self.transitions:
                possible_actions = list(self.transitions[s].keys())

            for a in possible_actions:
                q_value = 0
                if s in self.transitions and a in self.transitions[s]:
                    for prob, next_s in self.transitions[s][a]:
                        r = self.get_reward(s, a, next_s)
                        q_value += prob * (r + self.gamma * self.V[next_s])
                
                if q_value > max_q_value:
                    max_q_value = q_value
                    best_action = a
            
            # Update policy
            if best_action is not None:
                self.policy[s] = best_action
            
            if self.policy[s] != old_action:
                policy_stable = False
                
        return policy_stable

    def solve_policy_iteration(self):
        print("Starting Policy Iteration...")
        iterations = 0
        while True:
            iterations += 1
            self.policy_evaluation()
            stable = self.policy_improvement()
            if stable:
                print(f"Policy Iteration converged in {iterations} iterations.")
                break
        return self.policy, self.V

    def simulate_monte_carlo(self, start_state, num_episodes=1, horizon=100, policy=None):
        simulation_results = []

        print(f"\nStarting Monte Carlo Simulation ({'Policy' if policy else 'Random Walk'})...")

        for ep in range(num_episodes):
            current_state = start_state
            trajectory = []
            total_reward = 0.0
            discounted_return = 0.0
            
            for t in range(horizon):
                if current_state not in self.transitions or not self.transitions[current_state]:
                    break # Terminal state or dead end

                if policy:
                    action = policy.get(current_state)
                    if action not in self.transitions[current_state]:
                        break 
                else:
                    # Use Random Policy (Your snippet's logic)
                    action = random.choice(list(self.transitions[current_state].keys()))

                # 2. Sample Next State (Stochastic Transition)
                outcomes = self.transitions[current_state][action]
                probs = [o[0] for o in outcomes]
                states_out = [o[1] for o in outcomes]
                next_state = random.choices(states_out, weights=probs, k=1)[0]

                # 3. Get Reward
                reward = self.get_reward(current_state, action, next_state)

                # Record Data
                trajectory.append((current_state, action, reward))
                total_reward += reward
                discounted_return += (self.gamma ** t) * reward

                current_state = next_state
            
            simulation_results.append({
                "episode_index": ep + 1,
                "trajectory": trajectory,
                "total_reward": total_reward,
                "discounted_return": discounted_return
            })
            
        return simulation_results



# ==========================================
# Example Usage
# ==========================================
if __name__ == "__main__":
    states = ["S0", "S1", "S2", "S3", "S4", "S5"]
    actions = ["Right", "Down", "Left", "Up"]
    
    T = {
        "S0": {"Right": [(1.0, "S1")], "Down": [(1.0, "S3")]},
        "S1": {"Right": [(0.8, "S2"), (0.2, "S4")], "Left": [(1.0, "S0")], "Down": [(1.0, "S4")]}, 
        "S2": {}, 
        "S3": {"Right": [(1.0, "S4")], "Up": [(1.0, "S0")]},
        "S4": {"Right": [(1.0, "S5")], "Up": [(1.0, "S1")], "Left": [(1.0, "S3")]}, 
        "S5": {} 
    }
    
    R = {
        "S1": {"Right": {"S2": 10.0, "S4": 0.0}}, # +10 for reaching S2
        "S4": {"Right": {"S5": -10.0}}            # -10 for falling into S5
    }

    mdp = MDP(states, actions, T, R, gamma=0.9)
    
    optimal_policy, _ = mdp.solve_policy_iteration()
    
    print("\n--- 1. Simulating OPTIMAL Policy ---")
    results_opt = mdp.simulate_monte_carlo("S0", num_episodes=3, policy=optimal_policy)
    for res in results_opt:
        path = "->".join([x[0] for x in res['trajectory']])
        print(f"Return: {res['discounted_return']:.2f} | Path: {path}")

    print("\n--- 2. Simulating RANDOM Policy ---")
    results_rnd = mdp.simulate_monte_carlo("S0", num_episodes=3, policy=None)
    for res in results_rnd:
        path = "->".join([x[0] for x in res['trajectory']])
        print(f"Return: {res['discounted_return']:.2f} | Path: {path}")