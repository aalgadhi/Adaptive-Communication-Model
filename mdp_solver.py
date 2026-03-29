import random
from typing import Any, Dict, List, Tuple, Optional


class MDP:
    """
    Optimized finite MDP solver for sparse transition models.

    Keeps the original public API, but preprocesses the user-supplied nested
    dict structure into compact index-based lists so policy evaluation and
    policy improvement avoid repeated dictionary lookups and reward lookups.
    """

    def __init__(self, states, actions, transitions, rewards, gamma=0.98, epsilon=1e-6):
        self.states = list(states)
        self.actions = list(actions)
        self.transitions = transitions
        self.rewards = rewards
        self.gamma = float(gamma)
        self.epsilon = float(epsilon)

        # Public-facing containers retained for compatibility.
        self.V = {s: 0.0 for s in self.states}
        self.policy = {}
        self.greedy_policy = {}

        # Index maps.
        self.state_to_idx: Dict[Any, int] = {s: i for i, s in enumerate(self.states)}
        self.idx_to_state: List[Any] = self.states

        # For each state index, store available actions and sparse transition rows.
        # sa_rows[s_idx][a] = [(prob, next_idx, reward), ...]
        self.available_actions: List[List[Any]] = [[] for _ in self.states]
        self.sa_rows: List[Dict[Any, List[Tuple[float, int, float]]]] = [dict() for _ in self.states]
        self.sa_expected_reward: List[Dict[Any, float]] = [dict() for _ in self.states]

        # Build compact sparse structure once.
        fallback_action = self.actions[0] if self.actions else None
        for s_idx, s in enumerate(self.states):
            state_actions = transitions.get(s, {})
            if not state_actions:
                if fallback_action is not None:
                    self.policy[s] = fallback_action
                    self.greedy_policy[s] = fallback_action
                continue

            actions_here = list(state_actions.keys())
            self.available_actions[s_idx] = actions_here
            default_action = actions_here[0]
            self.policy[s] = default_action
            self.greedy_policy[s] = default_action

            reward_state = rewards.get(s, {})
            for a, outcomes in state_actions.items():
                row: List[Tuple[float, int, float]] = []
                expected_r = 0.0
                reward_action = reward_state.get(a, {})
                for prob, next_state in outcomes:
                    next_idx = self.state_to_idx[next_state]
                    r = reward_action.get(next_state, 0.0)
                    row.append((float(prob), next_idx, float(r)))
                    expected_r += float(prob) * float(r)
                self.sa_rows[s_idx][a] = row
                self.sa_expected_reward[s_idx][a] = expected_r

        # Dense arrays for faster internal iteration.
        self._V_arr: List[float] = [0.0] * len(self.states)
        self._policy_arr: List[Optional[Any]] = [self.policy.get(s) for s in self.states]

    def _sync_V_to_public(self):
        for idx, s in enumerate(self.states):
            self.V[s] = self._V_arr[idx]

    def _sync_policy_to_public(self):
        for idx, s in enumerate(self.states):
            a = self._policy_arr[idx]
            if a is not None:
                self.policy[s] = a

    def get_transition_prob(self, state, action, next_state):
        s_idx = self.state_to_idx.get(state)
        if s_idx is None:
            return 0.0
        for prob, ns_idx, _ in self.sa_rows[s_idx].get(action, []):
            if self.idx_to_state[ns_idx] == next_state:
                return prob
        return 0.0

    def get_reward(self, state, action, next_state):
        s_idx = self.state_to_idx.get(state)
        if s_idx is None:
            return 0.0
        for _, ns_idx, r in self.sa_rows[s_idx].get(action, []):
            if self.idx_to_state[ns_idx] == next_state:
                return r
        return 0.0

    def policy_evaluation(self, max_sweeps=10000):
        """Iterative in-place policy evaluation for the current policy."""
        n_states = len(self.states)
        gamma = self.gamma
        V = self._V_arr
        policy_arr = self._policy_arr
        sa_rows = self.sa_rows

        for _ in range(max_sweeps):
            delta = 0.0
            for s_idx in range(n_states):
                a = policy_arr[s_idx]
                if a is None:
                    continue
                row = sa_rows[s_idx].get(a)
                if not row:
                    new_v = 0.0
                else:
                    new_v = 0.0
                    for prob, next_idx, reward in row:
                        new_v += prob * (reward + gamma * V[next_idx])
                old_v = V[s_idx]
                V[s_idx] = new_v
                diff = abs(old_v - new_v)
                if diff > delta:
                    delta = diff
            if delta < self.epsilon:
                break

    def policy_improvement(self):
        """Greedily improves the policy using the current value function."""
        policy_stable = True
        gamma = self.gamma
        V = self._V_arr

        for s_idx in range(len(self.states)):
            actions_here = self.available_actions[s_idx]
            if not actions_here:
                continue

            old_action = self._policy_arr[s_idx]
            best_action = old_action
            best_q = float("-inf")

            for a in actions_here:
                q_val = 0.0
                for prob, next_idx, reward in self.sa_rows[s_idx][a]:
                    q_val += prob * (reward + gamma * V[next_idx])
                if q_val > best_q:
                    best_q = q_val
                    best_action = a

            if best_action != old_action:
                self._policy_arr[s_idx] = best_action
                policy_stable = False

        return policy_stable

    def solve_policy_iteration(self, max_iterations=1000, eval_max_sweeps=10000, verbose=True):
        """Solve the MDP using policy iteration."""
        if verbose:
            print("Starting Policy Iteration...")
        iterations = 0
        for _ in range(max_iterations):
            iterations += 1
            self.policy_evaluation(max_sweeps=eval_max_sweeps)
            stable = self.policy_improvement()
            if stable:
                break

        self._sync_V_to_public()
        self._sync_policy_to_public()
        if verbose:
            print(f"Policy Iteration converged in {iterations} iterations.")
        return self.policy, self.V

    def solve_greedy(self):
        """Creates a myopic greedy policy using expected immediate reward only."""
        for s_idx, s in enumerate(self.states):
            actions_here = self.available_actions[s_idx]
            if not actions_here:
                continue
            best_action = max(actions_here, key=lambda a: self.sa_expected_reward[s_idx][a])
            self.greedy_policy[s] = best_action
        return self.greedy_policy

    def simulate_monte_carlo(self, start_state, num_episodes=1, horizon=100, policy=None, seed=None, verbose=True):
        """Simulates episodes under a policy or random walk."""
        if seed is not None:
            random.seed(seed)

        simulation_results = []
        if verbose:
            print(f"\nStarting Monte Carlo Simulation ({'Policy' if policy else 'Random Walk'})...")

        start_idx = self.state_to_idx[start_state]

        for ep in range(num_episodes):
            current_idx = start_idx
            trajectory = []
            total_reward = 0.0
            discounted_return = 0.0

            for t in range(horizon):
                actions_here = self.available_actions[current_idx]
                current_state = self.idx_to_state[current_idx]
                if policy is not None:
                    action = policy.get(current_state)
                else:
                    action = random.choice(actions_here)

                row = self.sa_rows[current_idx][action]
                probs = [x[0] for x in row]
                chosen = random.choices(row, weights=probs, k=1)[0]
                _, next_idx, reward = chosen

                total_reward += reward
                trajectory.append((current_state, action, reward, total_reward))
                discounted_return += (self.gamma ** t) * reward
                current_idx = next_idx

            simulation_results.append(
                {
                    "episode_index": ep + 1,
                    "trajectory": trajectory,
                    "total_reward": total_reward,
                    "discounted_return": discounted_return,
                }
            )

        return simulation_results
