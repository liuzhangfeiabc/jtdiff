
import jittor as jt
from jittor import init
from jittor import nn
import enum
import math
import numpy as np
from .nn import mean_flat
from .losses import normal_kl, discretized_gaussian_log_likelihood

def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    if (schedule_name == 'linear'):
        scale = (1000 / num_diffusion_timesteps)
        beta_start = (scale * 0.0001)
        beta_end = (scale * 0.02)
        return np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif (schedule_name == 'cosine'):
        return betas_for_alpha_bar(num_diffusion_timesteps, (lambda t: (math.cos(((((t + 0.008) / 1.008) * math.pi) / 2)) ** 2)))
    else:
        raise NotImplementedError(f'unknown beta schedule: {schedule_name}')

def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    # '\n    Create a beta schedule that discretizes the given alpha_t_bar function,\n    which defines the cumulative product of (1-beta) over time from t = [0,1].\n\n    :param num_diffusion_timesteps: the number of betas to produce.\n    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and\n                      produces the cumulative product of (1-beta) up to that\n                      part of the diffusion process.\n    :param max_beta: the maximum beta to use; use values lower than 1 to\n                     prevent singularities.\n    '
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = (i / num_diffusion_timesteps)
        t2 = ((i + 1) / num_diffusion_timesteps)
        betas.append(min((1 - (alpha_bar(t2) / alpha_bar(t1))), max_beta))
    return np.array(betas)

class ModelMeanType(enum.Enum):
    # '\n    Which type of output the model predicts.\n    '
    PREVIOUS_X = enum.auto()
    START_X = enum.auto()
    EPSILON = enum.auto()

class ModelVarType(enum.Enum):
    # "\n    What is used as the model's output variance.\n\n    The LEARNED_RANGE option has been added to allow the model to predict\n    values between FIXED_SMALL and FIXED_LARGE, making its job easier.\n    "
    LEARNED = enum.auto()
    FIXED_SMALL = enum.auto()
    FIXED_LARGE = enum.auto()
    LEARNED_RANGE = enum.auto()

class LossType(enum.Enum):
    MSE = enum.auto()
    RESCALED_MSE = enum.auto()
    KL = enum.auto()
    RESCALED_KL = enum.auto()

    def is_vb(self):
        return ((self == LossType.KL) or (self == LossType.RESCALED_KL))

class GaussianDiffusion():
    # '\n    Utilities for training and sampling diffusion models.\n\n    Ported directly from here, and then adapted over time to further experimentation.\n    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py#L42\n\n    :param betas: a 1-D numpy array of betas for each diffusion timestep,\n                  starting at T and going to 1.\n    :param model_mean_type: a ModelMeanType determining what the model outputs.\n    :param model_var_type: a ModelVarType determining how variance is output.\n    :param loss_type: a LossType determining the loss function to use.\n    :param rescale_timesteps: if True, pass floating point timesteps into the\n                              model so that they are always scaled like in the\n                              original paper (0 to 1000).\n    '

    def __init__(self, *, betas, model_mean_type, model_var_type, loss_type, rescale_timesteps=False):
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        self.loss_type = loss_type
        self.rescale_timesteps = rescale_timesteps
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        assert (len(betas.shape) == 1), 'betas must be 1-D'
        assert ((betas > 0).all() and (betas <= 1).all())
        self.num_timesteps = int(betas.shape[0])
        alphas = (1.0 - betas)
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:(- 1)])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        assert (self.alphas_cumprod_prev.shape == (self.num_timesteps,))
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt((1.0 - self.alphas_cumprod))
        self.log_one_minus_alphas_cumprod = np.log((1.0 - self.alphas_cumprod))
        self.sqrt_recip_alphas_cumprod = np.sqrt((1.0 / self.alphas_cumprod))
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(((1.0 / self.alphas_cumprod) - 1))
        self.posterior_variance = ((betas * (1.0 - self.alphas_cumprod_prev)) / (1.0 - self.alphas_cumprod))
        self.posterior_log_variance_clipped = np.log(np.append(self.posterior_variance[1], self.posterior_variance[1:]))
        self.posterior_mean_coef1 = ((betas * np.sqrt(self.alphas_cumprod_prev)) / (1.0 - self.alphas_cumprod))
        self.posterior_mean_coef2 = (((1.0 - self.alphas_cumprod_prev) * np.sqrt(alphas)) / (1.0 - self.alphas_cumprod))

    def q_mean_variance(self, x_start, t):
        mean = (_extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start)
        variance = _extract_into_tensor((1.0 - self.alphas_cumprod), t, x_start.shape)
        log_variance = _extract_into_tensor(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return (mean, variance, log_variance)

    def q_sample(self, x_start, t, noise=None):
        if (noise is None):
            noise = jt.randn_like(x_start)
        assert (noise.shape == x_start.shape)
        return ((_extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start) + (_extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise))

    def q_posterior_mean_variance(self, x_start, x_t, t):
        assert (x_start.shape == x_t.shape)
        posterior_mean = ((_extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start) + (_extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t))
        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(self.posterior_log_variance_clipped, t, x_t.shape)
        assert (posterior_mean.shape[0] == posterior_variance.shape[0] == posterior_log_variance_clipped.shape[0] == x_start.shape[0])
        return (posterior_mean, posterior_variance, posterior_log_variance_clipped)

    def p_mean_variance(self, model, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None):
        if (model_kwargs is None):
            model_kwargs = {}
        (B, C) = x.shape[:2]
        assert (t.shape == (B,))
        # print('p_mean...')
        # print(t)
        model_output = model(x, self._scale_timesteps(t), **model_kwargs)
        # print("p_mean chkpoint")
        if (self.model_var_type in [ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE]):
            assert (model_output.shape == (B, (C * 2), *x.shape[2:]))
            (model_output, model_var_values) = jt.split(model_output, C, dim=1)
            if (self.model_var_type == ModelVarType.LEARNED):
                model_log_variance = model_var_values
                model_variance = jt.exp(model_log_variance)
            else:
                min_log = _extract_into_tensor(self.posterior_log_variance_clipped, t, x.shape)
                max_log = _extract_into_tensor(np.log(self.betas), t, x.shape)
                frac = ((model_var_values + 1) / 2)
                model_log_variance = ((frac * max_log) + ((1 - frac) * min_log))
                model_variance = jt.exp(model_log_variance)
        else:
            (model_variance, model_log_variance) = {ModelVarType.FIXED_LARGE: (np.append(self.posterior_variance[1], self.betas[1:]), np.log(np.append(self.posterior_variance[1], self.betas[1:]))), ModelVarType.FIXED_SMALL: (self.posterior_variance, self.posterior_log_variance_clipped)}[self.model_var_type]
            model_variance = _extract_into_tensor(model_variance, t, x.shape)
            model_log_variance = _extract_into_tensor(model_log_variance, t, x.shape)

        def process_xstart(x):
            if (denoised_fn is not None):
                x = denoised_fn(x)
            if clip_denoised:
                return x.clamp((- 1), 1)
            return x
        if (self.model_mean_type == ModelMeanType.PREVIOUS_X):
            pred_xstart = process_xstart(self._predict_xstart_from_xprev(x_t=x, t=t, xprev=model_output))
            model_mean = model_output
        elif (self.model_mean_type in [ModelMeanType.START_X, ModelMeanType.EPSILON]):
            if (self.model_mean_type == ModelMeanType.START_X):
                pred_xstart = process_xstart(model_output)
            else:
                pred_xstart = process_xstart(self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output))
            (model_mean, _, _) = self.q_posterior_mean_variance(x_start=pred_xstart, x_t=x, t=t)
        else:
            raise NotImplementedError(self.model_mean_type)
        assert (model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape)
        return {'mean': model_mean, 'variance': model_variance, 'log_variance': model_log_variance, 'pred_xstart': pred_xstart}

    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert (x_t.shape == eps.shape)
        return ((_extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t) - (_extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps))

    def _predict_xstart_from_xprev(self, x_t, t, xprev):
        assert (x_t.shape == xprev.shape)
        return ((_extract_into_tensor((1.0 / self.posterior_mean_coef1), t, x_t.shape) * xprev) - (_extract_into_tensor((self.posterior_mean_coef2 / self.posterior_mean_coef1), t, x_t.shape) * x_t))

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (((_extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t) - pred_xstart) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape))

    def _scale_timesteps(self, t):
        if self.rescale_timesteps:
            return (t.float() * (1000.0 / self.num_timesteps))
        return t

    def condition_mean(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
        gradient = cond_fn(x, self._scale_timesteps(t), **model_kwargs)
        new_mean = (p_mean_var['mean'].float() + (p_mean_var['variance'] * gradient.float()))
        return new_mean

    def condition_score(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
        eps = self._predict_eps_from_xstart(x, t, p_mean_var['pred_xstart'])
        eps = (eps - ((1 - alpha_bar).sqrt() * cond_fn(x, self._scale_timesteps(t), **model_kwargs)))
        out = p_mean_var.copy()
        out['pred_xstart'] = self._predict_xstart_from_eps(x, t, eps)
        (out['mean'], _, _) = self.q_posterior_mean_variance(x_start=out['pred_xstart'], x_t=x, t=t)
        return out

    def p_sample(self, model, x, t, clip_denoised=True, denoised_fn=None, cond_fn=None, model_kwargs=None):
        out = self.p_mean_variance(model, x, t, clip_denoised=clip_denoised, denoised_fn=denoised_fn, model_kwargs=model_kwargs)
        noise = jt.randn_like(x)
        nonzero_mask = (t != 0).float().view(((- 1), *([1] * (len(x.shape) - 1))))
        if (cond_fn is not None):
            out['mean'] = self.condition_mean(cond_fn, out, x, t, model_kwargs=model_kwargs)
        sample = (out['mean'] + ((nonzero_mask * jt.exp((0.5 * out['log_variance']))) * noise))
        return {'sample': sample, 'pred_xstart': out['pred_xstart']}

    def p_sample_loop(self, model, shape, noise=None, clip_denoised=True, denoised_fn=None, cond_fn=None, model_kwargs=None, device=None, progress=False):
        final = None
        for sample in self.p_sample_loop_progressive(model, shape, noise=noise, clip_denoised=clip_denoised, denoised_fn=denoised_fn, cond_fn=cond_fn, model_kwargs=model_kwargs, device=device, progress=progress):
            final = sample
        return final['sample']

    def p_sample_loop_progressive(self, model, shape, noise=None, clip_denoised=True, denoised_fn=None, cond_fn=None, model_kwargs=None, device=None, progress=False):
        if (device is None):
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if (noise is not None):
            img = noise
        else:
            img = jt.randn(*shape)
        indices = list(range(self.num_timesteps))[::(- 1)]
        if progress:
            from tqdm.auto import tqdm
            indices = tqdm(indices)
        for i in indices:
            t = jt.Var(([i] * shape[0]))
            with jt.no_grad():
                out = self.p_sample(model, img, t, clip_denoised=clip_denoised, denoised_fn=denoised_fn, cond_fn=cond_fn, model_kwargs=model_kwargs)
                (yield out)
                img = out['sample']

    def ddim_sample(self, model, x, t, clip_denoised=True, denoised_fn=None, cond_fn=None, model_kwargs=None, eta=0.0):
        # print('halfa')
        out = self.p_mean_variance(model, x, t, clip_denoised=clip_denoised, denoised_fn=denoised_fn, model_kwargs=model_kwargs)
        # print('halfb...')
        if (cond_fn is not None):
            out = self.condition_score(cond_fn, out, x, t, model_kwargs=model_kwargs)
        eps = self._predict_eps_from_xstart(x, t, out['pred_xstart'])
        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
        alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod_prev, t, x.shape)
        sigma = ((eta * jt.sqrt(((1 - alpha_bar_prev) / (1 - alpha_bar)))) * jt.sqrt((1 - (alpha_bar / alpha_bar_prev))))
        noise = jt.randn_like(x)
        mean_pred = ((out['pred_xstart'] * jt.sqrt(alpha_bar_prev)) + (jt.sqrt(((1 - alpha_bar_prev) - (sigma ** 2))) * eps))
        nonzero_mask = (t != 0).float().view(((- 1), *([1] * (len(x.shape) - 1))))
        sample = (mean_pred + ((nonzero_mask * sigma) * noise))
        return {'sample': sample, 'pred_xstart': out['pred_xstart']}

    def ddim_reverse_sample(self, model, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None, eta=0.0):
        assert (eta == 0.0), 'Reverse ODE only for deterministic path'
        out = self.p_mean_variance(model, x, t, clip_denoised=clip_denoised, denoised_fn=denoised_fn, model_kwargs=model_kwargs)
        eps = (((_extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x.shape) * x) - out['pred_xstart']) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x.shape))
        alpha_bar_next = _extract_into_tensor(self.alphas_cumprod_next, t, x.shape)
        mean_pred = ((out['pred_xstart'] * jt.sqrt(alpha_bar_next)) + (jt.sqrt((1 - alpha_bar_next)) * eps))
        return {'sample': mean_pred, 'pred_xstart': out['pred_xstart']}

    def ddim_sample_loop(self, model, shape, noise=None, clip_denoised=True, denoised_fn=None, cond_fn=None, model_kwargs=None, device=None, progress=False, eta=0.0):
        final = None
        # final = self.ddim_sample_loop_progressive(model, shape, noise=noise, clip_denoised=clip_denoised, denoised_fn=denoised_fn, cond_fn=cond_fn, model_kwargs=model_kwargs, device=device, progress=progress, eta=eta)
        for sample in self.ddim_sample_loop_progressive(model, shape, noise=noise, clip_denoised=clip_denoised, denoised_fn=denoised_fn, cond_fn=cond_fn, model_kwargs=model_kwargs, device=device, progress=progress, eta=eta):
            # print("begin2")
            final = sample
            jt.clean()
            # print("end2")
        return final['sample']

    def ddim_sample_loop_progressive(self, model, shape, noise=None, clip_denoised=True, denoised_fn=None, cond_fn=None, model_kwargs=None, device=None, progress=False, eta=0.0):
        # print("ddim_sample_loop_progressive")
        if (device is None):
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if (noise is not None):
            img = noise
        else:
            img = jt.randn(*shape, )
        indices = list(range(self.num_timesteps))[::(- 1)]
        if progress:
            from tqdm.auto import tqdm
            indices = tqdm(indices)
        for i in indices:
            t = jt.array(([i] * shape[0]))
            # print('loop_progressive...')
            with jt.no_grad():
                out = self.ddim_sample(model, img, t, clip_denoised=clip_denoised, denoised_fn=denoised_fn, cond_fn=cond_fn, model_kwargs=model_kwargs, eta=eta)
                yield out
                img = out['sample']

    def _vb_terms_bpd(self, model, x_start, x_t, t, clip_denoised=True, model_kwargs=None):
        # "\n        Get a term for the variational lower-bound.\n\n        The resulting units are bits (rather than nats, as one might expect).\n        This allows for comparison to other papers.\n\n        :return: a dict with the following keys:\n                 - 'output': a shape [N] tensor of NLLs or KLs.\n                 - 'pred_xstart': the x_0 predictions.\n        "
        (true_mean, _, true_log_variance_clipped) = self.q_posterior_mean_variance(x_start=x_start, x_t=x_t, t=t)
        out = self.p_mean_variance(model, x_t, t, clip_denoised=clip_denoised, model_kwargs=model_kwargs)
        kl = normal_kl(true_mean, true_log_variance_clipped, out['mean'], out['log_variance'])
        kl = (mean_flat(kl) / np.log(2.0))
        decoder_nll = (- discretized_gaussian_log_likelihood(x_start, means=out['mean'], log_scales=(0.5 * out['log_variance'])))
        assert (decoder_nll.shape == x_start.shape)
        decoder_nll = (mean_flat(decoder_nll) / np.log(2.0))
        output = jt.where((t == 0), decoder_nll, kl)
        return {'output': output, 'pred_xstart': out['pred_xstart']}

    def training_losses(self, model, x_start, t, model_kwargs=None, noise=None):
        # '\n        Compute training losses for a single timestep.\n\n        :param model: the model to evaluate loss on.\n        :param x_start: the [N x C x ...] tensor of inputs.\n        :param t: a batch of timestep indices.\n        :param model_kwargs: if not None, a dict of extra keyword arguments to\n            pass to the model. This can be used for conditioning.\n        :param noise: if specified, the specific Gaussian noise to try to remove.\n        :return: a dict with the key "loss" containing a tensor of shape [N].\n                 Some mean or variance settings may also have other keys.\n        '
        if (model_kwargs is None):
            model_kwargs = {}
        if (noise is None):
            noise = jt.randn_like(x_start)
        x_t = self.q_sample(x_start, t, noise=noise)
        terms = {}
        if ((self.loss_type == LossType.KL) or (self.loss_type == LossType.RESCALED_KL)):
            terms['loss'] = self._vb_terms_bpd(model=model, x_start=x_start, x_t=x_t, t=t, clip_denoised=False, model_kwargs=model_kwargs)['output']
            if (self.loss_type == LossType.RESCALED_KL):
                terms['loss'] *= self.num_timesteps
        elif ((self.loss_type == LossType.MSE) or (self.loss_type == LossType.RESCALED_MSE)):
            model_output = model(x_t, self._scale_timesteps(t), **model_kwargs)
            if (self.model_var_type in [ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE]):
                (B, C) = x_t.shape[:2]
                assert (model_output.shape == (B, (C * 2), *x_t.shape[2:]))
                (model_output, model_var_values) = jt.split(model_output, C, dim=1)
                frozen_out = jt.contrib.concat([model_output.detach(), model_var_values], dim=1)
                terms['vb'] = self._vb_terms_bpd(model=(lambda *args, r=frozen_out: r), x_start=x_start, x_t=x_t, t=t, clip_denoised=False)['output']
                if (self.loss_type == LossType.RESCALED_MSE):
                    terms['vb'] *= (self.num_timesteps / 1000.0)
            target = {ModelMeanType.PREVIOUS_X: self.q_posterior_mean_variance(x_start=x_start, x_t=x_t, t=t)[0], ModelMeanType.START_X: x_start, ModelMeanType.EPSILON: noise}[self.model_mean_type]
            assert (model_output.shape == target.shape == x_start.shape)
            terms['mse'] = mean_flat(((target - model_output) ** 2))
            if ('vb' in terms):
                terms['loss'] = (terms['mse'] + terms['vb'])
            else:
                terms['loss'] = terms['mse']
        else:
            raise NotImplementedError(self.loss_type)
        return terms

    def _prior_bpd(self, x_start):
        # "\n        Get the prior KL term for the variational lower-bound, measured in\n        bits-per-dim.\n\n        This term can't be optimized, as it only depends on the encoder.\n\n        :param x_start: the [N x C x ...] tensor of inputs.\n        :return: a batch of [N] KL values (in bits), one per batch element.\n        "
        batch_size = x_start.shape[0]
        t = jt.tensor(([(self.num_timesteps - 1)] * batch_size), device=x_start.device)
        (qt_mean, _, qt_log_variance) = self.q_mean_variance(x_start, t)
        kl_prior = normal_kl(mean1=qt_mean, logvar1=qt_log_variance, mean2=0.0, logvar2=0.0)
        return (mean_flat(kl_prior) / np.log(2.0))

    def calc_bpd_loop(self, model, x_start, clip_denoised=True, model_kwargs=None):
        # '\n        Compute the entire variational lower-bound, measured in bits-per-dim,\n        as well as other related quantities.\n\n        :param model: the model to evaluate loss on.\n        :param x_start: the [N x C x ...] tensor of inputs.\n        :param clip_denoised: if True, clip denoised samples.\n        :param model_kwargs: if not None, a dict of extra keyword arguments to\n            pass to the model. This can be used for conditioning.\n\n        :return: a dict containing the following keys:\n                 - total_bpd: the total variational lower-bound, per batch element.\n                 - prior_bpd: the prior term in the lower-bound.\n                 - vb: an [N x T] tensor of terms in the lower-bound.\n                 - xstart_mse: an [N x T] tensor of x_0 MSEs for each timestep.\n                 - mse: an [N x T] tensor of epsilon MSEs for each timestep.\n        '
        device = x_start.device
        batch_size = x_start.shape[0]
        vb = []
        xstart_mse = []
        mse = []
        for t in list(range(self.num_timesteps))[::(- 1)]:
            t_batch = jt.tensor(([t] * batch_size), device=device)
            noise = jt.randn_like(x_start)
            x_t = self.q_sample(x_start=x_start, t=t_batch, noise=noise)
            with jt.no_grad():
                out = self._vb_terms_bpd(model, x_start=x_start, x_t=x_t, t=t_batch, clip_denoised=clip_denoised, model_kwargs=model_kwargs)
            vb.append(out['output'])
            xstart_mse.append(mean_flat(((out['pred_xstart'] - x_start) ** 2)))
            eps = self._predict_eps_from_xstart(x_t, t_batch, out['pred_xstart'])
            mse.append(mean_flat(((eps - noise) ** 2)))
        vb = jt.stack(vb, dim=1)
        xstart_mse = jt.stack(xstart_mse, dim=1)
        mse = jt.stack(mse, dim=1)
        prior_bpd = self._prior_bpd(x_start)
        total_bpd = (vb.sum(dim=1) + prior_bpd)
        return {'total_bpd': total_bpd, 'prior_bpd': prior_bpd, 'vb': vb, 'xstart_mse': xstart_mse, 'mse': mse}

def _extract_into_tensor(arr, timesteps, broadcast_shape):
    # '\n    Extract values from a 1-D numpy array for a batch of indices.\n\n    :param arr: the 1-D numpy array.\n    :param timesteps: a tensor of indices into the array to extract.\n    :param broadcast_shape: a larger shape of K dimensions with the batch\n                            dimension equal to the length of timesteps.\n    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.\n    '
    res = jt.array(arr)[timesteps].float()
    while (len(res.shape) < len(broadcast_shape)):
        res = res[(..., None)]
    return res.expand(broadcast_shape)

