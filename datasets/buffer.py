import numpy as np

def atleast_2d(x):
    while x.ndim < 2:
        x = np.expand_dims(x, axis=-1)
    return x

class ReplayBuffer:
    def __init__(self, termination_penalty):
        self._dict = {
            'path_lengths': [],
        }
        self._count = 0
        self.termination_penalty = termination_penalty

    def __repr__(self):
        return '[ datasets/buffer ] Info:\n' + '\n'.join(
            f'    {key}: [{np.sum(self.path_lengths)}, {np.shape(val[0])[-1]}]'
            for key, val in self.items()
        )

    def __getitem__(self, key):
        return self._dict[key]

    def __setitem__(self, key, val):
        self._dict[key] = val
        self._add_attributes()

    @property
    def n_episodes(self):
        return self._count

    @property
    def n_steps(self):
        return sum(self['path_lengths'])

    def _add_keys(self, path):
        if hasattr(self, 'keys'):
            return
        self.keys = list(path.keys())

    def _add_attributes(self):
        '''
            can access fields with `buffer.observations`
            instead of `buffer['observations']`
        '''
        for key, val in self._dict.items():
            setattr(self, key, val)

    def items(self):
        return {k: v for k, v in self._dict.items()
                if k != 'path_lengths'}.items()

    def _add_value_for_key(self):
        for key in self.keys:
            if key not in self._dict.keys():
                self._dict[key] = []

    def add_path(self, path):
        path_length = len(path['observations'])
        if path['terminals'].any():
            assert (path['terminals'][-1] == True) and (not path['terminals'][:-1].any())
        ## if first path added, set keys based on contents
        self._add_keys(path)
        self._add_value_for_key()

        ## add tracked keys in path
        for key in self.keys:
            array = atleast_2d(path[key])
            self._dict[key].append(array)

        try:
            ## penalize early termination
            if path['terminals'].any() and self.termination_penalty is not None:
                assert not path['timeouts'].any(), 'Penalized a timeout episode for early termination'
                self._dict['rewards'][self._count][-1] += self.termination_penalty
        except:
            pass

        ## record path length
        self._dict['path_lengths'].append(path_length)

        ## increment path counter
        self._count += 1

    def truncate_path(self, path_ind, step):
        old = self._dict['path_lengths'][path_ind]
        new = min(step, old)
        self._dict['path_lengths'][path_ind] = new

    def finalize(self):
        self._add_attributes()
        # print(f'[ datasets/buffer ] Finalized replay buffer | {self._count} episodes')

class ReturnReplayBuffer(ReplayBuffer):
    def __init__(self, argus, termination_penalty, discounts, max_path_length):
        super().__init__(termination_penalty=termination_penalty)
        self._dict = {
            'path_lengths': [],
            'discounted_returns': [],
        }
        self.argus = argus
        self._count = 0
        self.termination_penalty = termination_penalty
        self.discounts = discounts ** np.arange(max_path_length)
        self.max_return, self.min_return = -9999, 99999

    def add_path(self, path):
        path_length = len(path['observations'])
        if path['terminals'].any():
            assert (path['terminals'][-1] == True) and (not path['terminals'][:-1].any())
        ## if first path added, set keys based on contents
        self._add_keys(path)
        self._add_value_for_key()
        ## add tracked keys in path
        for key in self.keys:
            array = atleast_2d(path[key])
            self._dict[key].append(array)
        ## penalize early termination
        try:
            if path['terminals'].any() and self.termination_penalty is not None:
                assert not path['timeouts'].any(), 'Penalized a timeout episode for early termination'
                self._dict['rewards'][self._count][-1] += self.termination_penalty
        except:
            pass
        ## record path length
        self._dict['path_lengths'].append(path_length)
        path_returns = []
        for start_i in range(path_length):
            rewards = path["rewards"][start_i:]
            # returns = (self.discounts[:len(rewards)] * np.squeeze(rewards)).sum()
            returns = rewards.sum()
            if returns > self.max_return: self.max_return = returns
            if returns < self.min_return: self.min_return = returns
            # returns = np.array([returns / self.returns_scale], dtype=np.float32)
            path_returns.append(returns)
        self._dict['discounted_returns'].append(np.reshape(np.array(path_returns), [-1, 1]))

        ## increment path counter
        self._count += 1

    def return_normalization(self):
        for path_i in range(len(self._dict['discounted_returns'])):
            self._dict['discounted_returns'][path_i] = (self._dict['discounted_returns'][path_i] - self.min_return) / (self.max_return - self.min_return)

    def finalize(self):
        self.return_normalization()
        self._add_attributes()
        # print(f'[ datasets/buffer ] Finalized replay buffer | {self._count} episodes')