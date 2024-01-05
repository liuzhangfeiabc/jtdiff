
import jittor as jt
from jittor import init
from jittor import nn
from abc import ABC, abstractmethod
import numpy as np
import jittor.distributions as dist

def create_named_schedule_sampler(name, diffusion):
    # '\n    Create a ScheduleSampler from a library of pre-defined samplers.\n\n    :param name: the name of the sampler.\n    :param diffusion: the diffusion object to sample for.\n    '
    if (name == 'uniform'):
        return UniformSampler(diffusion)
    elif (name == 'loss-second-moment'):
        return LossSecondMomentResampler(diffusion)
    else:
        raise NotImplementedError(f'unknown schedule sampler: {name}')

class ScheduleSampler(ABC):
    # "\n    A distribution over timesteps in the diffusion process, intended to reduce\n    variance of the objective.\n\n    By default, samplers perform unbiased importance sampling, in which the\n    objective's mean is unchanged.\n    However, subclasses may override sample() to change how the resampled\n    terms are reweighted, allowing for actual changes in the objective.\n    "

    @abstractmethod
    def weights(self):
        """
        Get a numpy array of weights, one per diffusion step.

        The weights needn't be normalized, but must be positive.
        """

    def sample(self, batch_size, device):
        # '\n        Importance-sample timesteps for a batch.\n\n        :param batch_size: the number of timesteps.\n        :param device: the torch device to save to.\n        :return: a tuple (timesteps, weights):\n                 - timesteps: a tensor of timestep indices.\n                 - weights: a tensor of weights to scale the resulting losses.\n        '
        w = self.weights()
        p = (w / np.sum(w))
        indices_np = np.random.choice(len(p), size=(batch_size,), p=p)
        indices = jt.array(indices_np).long()
        weights_np = (1 / (len(p) * p[indices_np]))
        weights = jt.array(weights_np).float()
        return (indices, weights)

class UniformSampler(ScheduleSampler):

    def __init__(self, diffusion):
        self.diffusion = diffusion
        self._weights = np.ones([diffusion.num_timesteps])

    def weights(self):
        return self._weights

class LossAwareSampler(ScheduleSampler):

    def update_with_local_losses(self, local_ts, local_losses):
        '\n        Update the reweighting using losses from a model.\n\n        Call this method from each rank with a batch of timesteps and the\n        corresponding losses for each of those timesteps.\n        This method will perform synchronization to make sure all of the ranks\n        maintain the exact same reweighting.\n\n        :param local_ts: an integer Tensor of timesteps.\n        :param local_losses: a 1D Tensor of losses.\n        '
        batch_sizes = [jt.tensor([0], dtype=jt.int32, device=local_ts.device) for _ in range(dist.get_world_size())]
        dist.all_gather(batch_sizes, jt.tensor([len(local_ts)], dtype=jt.int32, device=local_ts.device))
        batch_sizes = [x.item() for x in batch_sizes]
        max_bs = max(batch_sizes)
        timestep_batches = [jt.zeros(max_bs).to(local_ts) for bs in batch_sizes]
        loss_batches = [jt.zeros(max_bs).to(local_losses) for bs in batch_sizes]
        dist.all_gather(timestep_batches, local_ts)
        dist.all_gather(loss_batches, local_losses)
        timesteps = [x.item() for (y, bs) in zip(timestep_batches, batch_sizes) for x in y[:bs]]
        losses = [x.item() for (y, bs) in zip(loss_batches, batch_sizes) for x in y[:bs]]
        self.update_with_all_losses(timesteps, losses)

    @abstractmethod
    def update_with_all_losses(self, ts, losses):
        '\n        Update the reweighting using losses from a model.\n\n        Sub-classes should override this method to update the reweighting\n        using losses from the model.\n\n        This method directly updates the reweighting without synchronizing\n        between workers. It is called by update_with_local_losses from all\n        ranks with identical arguments. Thus, it should have deterministic\n        behavior to maintain state across workers.\n\n        :param ts: a list of int timesteps.\n        :param losses: a list of float losses, one per timestep.\n        '

class LossSecondMomentResampler(LossAwareSampler):

    def __init__(self, diffusion, history_per_term=10, uniform_prob=0.001):
        self.diffusion = diffusion
        self.history_per_term = history_per_term
        self.uniform_prob = uniform_prob
        self._loss_history = np.zeros([diffusion.num_timesteps, history_per_term], dtype=np.float64)
        self._loss_counts = np.zeros([diffusion.num_timesteps], dtype=np.int)

    def weights(self):
        if (not self._warmed_up()):
            return np.ones([self.diffusion.num_timesteps], dtype=np.float64)
        weights = np.sqrt(np.mean((self._loss_history ** 2), axis=(- 1)))
        weights /= np.sum(weights)
        weights *= (1 - self.uniform_prob)
        weights += (self.uniform_prob / len(weights))
        return weights

    def update_with_all_losses(self, ts, losses):
        for (t, loss) in zip(ts, losses):
            if (self._loss_counts[t] == self.history_per_term):
                self._loss_history[t, :(- 1)] = self._loss_history[t, 1:]
                self._loss_history[(t, (- 1))] = loss
            else:
                self._loss_history[(t, self._loss_counts[t])] = loss
                self._loss_counts[t] += 1

    def _warmed_up(self):
        return (self._loss_counts == self.history_per_term).all()

