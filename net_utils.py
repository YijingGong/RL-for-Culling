"""
net_utils.py  —  backend-agnostic access to a trained DQN policy.

Option A of the production-level visualization plan: the neural network
(saved to <name>_model.pth) is the SOURCE OF TRUTH for the continuous
production level. This module exposes the network as a plain callable
Q-function so the visualization code can query it at ANY production level,
not just the coarse M_GRID baked into the .pkl.

It works in two environments:
  * If PyTorch is installed (your training/eval machine) it loads the
    checkpoint the normal way and reuses the exact input scaling.
  * If PyTorch is NOT installed (e.g. a lightweight plotting box) it falls
    back to a pure-numpy loader + forward pass. The numpy path has been
    verified to reproduce the trained network's Q-values to the cent.

The input scaling MUST match training exactly:
    [parity/12, mac/20, mip/9, disease, (production_level - 1)/0.11]
"""
import os
import numpy as np

PROD_SD_NORM = 0.11  # matches PRODUCTION_MULT_SD used to standardize the input


def _scale(state):
    """state = (parity, mac, mip, disease, production_level) -> scaled 5-vector."""
    parity, mac, mip, disease, prod = state
    return np.array([parity / 12.0, mac / 20.0, mip / 9.0,
                     float(disease), (prod - 1.0) / PROD_SD_NORM], dtype=np.float32)


# ───────────────────────── numpy-only checkpoint loader ─────────────────────
def _load_state_dict_numpy(pth_path):
    """Read a torch .pth (zip) checkpoint's policy_net_state_dict without torch."""
    import zipfile, pickle, io
    z = zipfile.ZipFile(pth_path)
    root = z.namelist()[0].split('/')[0]
    raw = {n.split('/')[-1]: z.read(n) for n in z.namelist()
           if '/data/' in n and n.split('/')[-1].isdigit()}
    dtmap = {'FloatStorage': np.float32, 'DoubleStorage': np.float64,
             'LongStorage': np.int64, 'IntStorage': np.int32, 'HalfStorage': np.float16}

    class _Storage:
        def __init__(self, key, dtype): self.key, self.dtype = key, dtype

    class U(pickle.Unpickler):
        def find_class(self, mod, name):
            if name == '_rebuild_tensor_v2':
                return self._rebuild
            if name == 'OrderedDict':
                from collections import OrderedDict
                return OrderedDict
            try:
                return super().find_class(mod, name)
            except Exception:
                return lambda *a, **k: None

        @staticmethod
        def _rebuild(storage, offset, size, stride, requires_grad, hooks, *a):
            arr = np.frombuffer(raw[storage.key], dtype=storage.dtype).copy()
            n = int(np.prod(size)) if size else arr.size
            arr = arr[offset:offset + n]
            return arr.reshape(size) if size else arr

        def persistent_load(self, pid):
            typ, key = pid[1], str(pid[2])
            tname = typ if isinstance(typ, str) else getattr(typ, '__name__', str(typ))
            return _Storage(key, dtmap.get(tname.split('.')[-1], np.float32))

    ck = U(io.BytesIO(z.read(root + '/data.pkl'))).load()
    return ck['policy_net_state_dict']


class QFunction:
    """Callable Q-function over continuous states.

    q(state)        -> np.array([Q_keep, Q_replace])
    value(state)    -> Q_keep - Q_replace   (retention value; >0 => keep)
    q_grid(states)  -> (N,2) array for a list/array of states (vectorized)
    """

    def __init__(self, pth_path):
        self.pth_path = pth_path
        self._torch = None
        try:
            import torch  # noqa
            from dqn_learning import DQN, state_to_tensor
            ck = torch.load(pth_path, weights_only=False)
            net = DQN(state_dim=5, hidden_dim=64, action_dim=2)
            net.load_state_dict(ck['policy_net_state_dict'])
            net.eval()
            self._torch = (torch, net, state_to_tensor)
        except Exception:
            sd = _load_state_dict_numpy(pth_path)
            self.W1, self.b1 = np.asarray(sd['fc1.weight']), np.asarray(sd['fc1.bias'])
            self.W2, self.b2 = np.asarray(sd['fc2.weight']), np.asarray(sd['fc2.bias'])
            self.W3, self.b3 = np.asarray(sd['fc3.weight']), np.asarray(sd['fc3.bias'])

    def q(self, state):
        if self._torch is not None:
            torch, net, s2t = self._torch
            with torch.no_grad():
                return net(s2t(state).unsqueeze(0)).squeeze(0).numpy()
        x = _scale(state)
        h = np.maximum(self.W1 @ x + self.b1, 0)
        h = np.maximum(self.W2 @ h + self.b2, 0)
        return self.W3 @ h + self.b3

    def value(self, state):
        qk, qr = self.q(state)
        return float(qk - qr)

    def q_grid(self, states):
        return np.array([self.q(s) for s in states])


class EnsembleQFunction:
    """Seed-averaged Q-function: averages Q(keep) and Q(replace) across the per-seed
    networks. Because value() = Q(keep) - Q(replace), averaging the per-action Q across
    seeds is identical to averaging the keep-replace difference across seeds — matching
    how visualize_scenario.py builds Figures 2/3 (mean of the difference over 5 seeds)."""

    def __init__(self, members):
        self.members = members          # list[QFunction]

    def q(self, state):
        return np.mean([m.q(state) for m in self.members], axis=0)

    def value(self, state):
        qk, qr = self.q(state)
        return float(qk - qr)

    def q_grid(self, states):
        return np.array([self.q(s) for s in states])


def find_qfunc(collected_dir, prefix, scenario, seed):
    """Locate <prefix>_<scenario>_seed<seed>_model.pth in collected_dir; return QFunction or None."""
    for name in (f"{prefix}_{scenario}_seed{seed}_model.pth",
                 f"{prefix}_{scenario}_{seed}_model.pth"):
        p = os.path.join(collected_dir, name)
        if os.path.exists(p):
            return QFunction(p)
    return None


def find_qfunc_ensemble(collected_dir, prefix, scenario, seeds):
    """Load every available seed for a scenario and return a seed-averaged Q-function.
    Falls back to a single QFunction if only one seed is present, or None if none are."""
    members = []
    for seed in seeds:
        qf = find_qfunc(collected_dir, prefix, scenario, seed)
        if qf is not None:
            members.append(qf)
    if not members:
        return None
    return members[0] if len(members) == 1 else EnsembleQFunction(members)