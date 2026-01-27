import numpy as np

class MonteCarlo:
    def __init__(self, num_channels, num_states, num_coding_schemes,
                 alpha_freq=1.0, alpha_mcs=0.3, block_penalty=20.0, eps=1e-6, seed=None):
        assert num_states == 2, "This minimal version supports only two qualities: Poor/Good."
        self.rng = np.random.default_rng(seed)
        self.num_channels = num_channels
        self.num_states = num_states
        self.num_coding_schemes = num_coding_schemes
        self.eps = float(eps)

        self.state = dict()
        self.state["jamming"] = [False] * self.num_channels         
        self.state["channels_state"] = ["Poor"] * self.num_channels 
        self.prev_freq = None
        self.prev_mcs  = None

        self.actions = dict()
        self.actions["coding_schemes"] = [i for i in range(1, num_coding_schemes + 1)]
        self.actions["frequencies"]    = [i for i in range(1, num_channels + 1)]

        self.rates_per_mcs = [0.5 + 0.5*(i-1) for i in self.actions["coding_schemes"]]  # [0.5,1.0,1.5,...]
        self.quality_scale = {"Poor": 0.3, "Good": 1.0}
        self.alpha_freq = float(alpha_freq)   # cost of switching frequency
        self.alpha_mcs  = float(alpha_mcs)    # cost of switching MCS (constant)
        self.block_penalty = float(block_penalty)

        self.t = 0
        self.total_cost = 0.0
        self.path = []   # list of (t, jam[], qual[], action, rate, cost)

    def _rate(self, freq, mcs):
        f_idx = freq - 1
        m_idx = mcs - 1
        if self.state["jamming"][f_idx]:
            return 0.0
        base = self.rates_per_mcs[m_idx]
        q    = self.quality_scale[self.state["channels_state"][f_idx]]
        return base * q

    def _cost(self, freq, mcs, r):
        # delay term
        delay = self.block_penalty if r <= 0 else 1.0 / max(r, self.eps)
        # switching terms
        c_freq = self.alpha_freq if (self.prev_freq is not None and freq != self.prev_freq) else 0.0
        c_mcs  = self.alpha_mcs  if (self.prev_mcs  is not None and mcs  != self.prev_mcs)  else 0.0
        return delay + c_freq + c_mcs

    def update_state(self):
        # --- randomly evolve environment (i.i.d. each step) ---
        self.state["jamming"] = [self.rng.random() < 0.3 for _ in range(self.num_channels)]
        self.state["channels_state"] = ["Good" if self.rng.random() < 0.6 else "Poor"
                                        for _ in range(self.num_channels)]

        freq = int(self.rng.choice(self.actions["frequencies"]))
        mcs  = int(self.rng.choice(self.actions["coding_schemes"]))
        r = self._rate(freq, mcs)
        c = self._cost(freq, mcs, r)

        self.path.append((self.t,
                          self.state["jamming"][:],
                          self.state["channels_state"][:],
                          (freq, mcs), r, c))
        self.total_cost += c
        self.prev_freq = freq
        self.prev_mcs  = mcs
        self.t += 1

    def print_sample_path(self):
        print(f"{'t':>2} | jammed | quality | (f,m) | rate | cost")
        for t, jam, qual, act, r, c in self.path:
            print(f"{t:2d} | {jam} | {qual} | {act} | {r:4.2f} | {c:5.2f}")
        print(f"Total cost = {self.total_cost:.2f}")

if __name__ == "__main__":
    number_of_sample_paths = 10
    number_of_iters = 20
    for k in range(number_of_sample_paths):
        path = MonteCarlo(2, 2, 3, alpha_freq=1.0, alpha_mcs=0.3, seed=7 + k)  # <-- vary seed
        for _ in range(number_of_iters):
            path.update_state()
        path.print_sample_path()
        print("-" * 60)

