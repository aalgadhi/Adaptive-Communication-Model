import random
import time

class MDP:
    def __init__(self, states, actions, transitions, rewards, gamma=0.99, epsilon=1e-6):
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
        self.greedy_policy = {s: self.actions[0] for s in self.states}


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
        """Iteratively updates the value function V(s) for the current policy."""
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
        """Greedily updates the policy based on the current value function V(s)."""
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
        """Solves the MDP using the Policy Iteration algorithm."""
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
    
    # def solve_greedy(self):
    #     """Creates a Greedy Policy"""
    #     print("Greedy Policy...")

    #     for s in self.states:
    #         best_action = None
    #         max_r = float('-inf')
    #         possible_actions = self.actions

    #         if s in self.transitions:
    #             possible_actions = list(self.transitions[s].keys())

    #         for a in possible_actions:
    #             if s in self.transitions and a in self.transitions[s]:
    #                 for prob, next_s in self.transitions[s][a]:
    #                     if next_s[:2] == s[:2]: # Conditioning the next state jammers and channels to be the same
    #                         r = self.get_reward(s, a, next_s)

    #                         if r > max_r:
    #                             max_r = r
    #                             best_action = a
                        
    #         # Update policy
    #         if best_action is not None:
    #             self.greedy_policy[s] = best_action

    #     return self.greedy_policy
    

    def solve_greedy(self):
        """Creates a Myopic (Short-term) Greedy Policy."""
        for s in self.states:
            best_action = None
            max_expected_r = float('-inf')
            
            # We only look at actions available for this specific state
            actions_to_try = self.transitions.get(s, {}).keys()

            for a in actions_to_try:
                # Expected Reward = Sum(P(s'|s,a) * R(s,a,s'))
                current_expected_r = sum(
                    prob * self.get_reward(s, a, ns) 
                    for prob, ns in self.transitions[s][a]
                )

                if current_expected_r > max_expected_r:
                    max_expected_r = current_expected_r
                    best_action = a
            
            if best_action:
                self.greedy_policy[s] = best_action
        return self.greedy_policy


    def simulate_monte_carlo(self, start_state, num_episodes=1, horizon=100, policy=None):
        """Simulates the environment to evaluate policy performance or random walks."""
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
                    # Use Random Policy
                    action = random.choice(list(self.transitions[current_state].keys()))

                # Sample Next State (Stochastic Transition)
                outcomes = self.transitions[current_state][action]
                probs = [o[0] for o in outcomes]
                states_out = [o[1] for o in outcomes]
                next_state = random.choices(states_out, weights=probs, k=1)[0]

                # Get Reward
                reward = self.get_reward(current_state, action, next_state)
                # Record Data
                total_reward += reward
                trajectory.append((current_state, action, reward, total_reward))
                discounted_return += (self.gamma ** t) * reward

                current_state = next_state
            
            simulation_results.append({
                "episode_index": ep + 1,
                "trajectory": trajectory,
                "total_reward": total_reward,
                "discounted_return": discounted_return
            })
            
        return simulation_results
