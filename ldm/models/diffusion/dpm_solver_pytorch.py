import torch
import torch.nn.functional as F
import math


class NoiseScheduleVP:
    def __init__(
            self,
            schedule='discrete',
            betas=None,
            alphas_cumprod=None,
            continuous_beta_0=0.1,
            continuous_beta_1=20.,
            dtype=torch.float32,
        ):
        """Create a wrapper class for the forward SDE (VP type).

        ***
        Update: We support discrete-time diffusion models by implementing a picewise linear interpolation for log_alpha_t.
                We recommend to use schedule='discrete' for the discrete-time diffusion models, especially for high-resolution images.
        ***

        The forward SDE ensures that the condition distribution q_{t|0}(x_t | x_0) = N ( alpha_t * x_0, sigma_t^2 * I ).
        We further define lambda_t = log(alpha_t) - log(sigma_t), which is the half-logSNR (described in the DPM-Solver paper).
        Therefore, we implement the functions for computing alpha_t, sigma_t and lambda_t. For t in [0, T], we have:

            log_alpha_t = self.marginal_log_mean_coeff(t)
            sigma_t = self.marginal_std(t)
            lambda_t = self.marginal_lambda(t)

        Moreover, as lambda(t) is an invertible function, we also support its inverse function:

            t = self.inverse_lambda(lambda_t)

        ===============================================================

        We support both discrete-time DPMs (trained on n = 0, 1, ..., N-1) and continuous-time DPMs (trained on t in [t_0, T]).

        1. For discrete-time DPMs:

            For discrete-time DPMs trained on n = 0, 1, ..., N-1, we convert the discrete steps to continuous time steps by:
                t_i = (i + 1) / N
            e.g. for N = 1000, we have t_0 = 1e-3 and T = t_{N-1} = 1.
            We solve the corresponding diffusion ODE from time T = 1 to time t_0 = 1e-3.

            Args:
                betas: A `torch.Tensor`. The beta array for the discrete-time DPM. (See the original DDPM paper for details)
                alphas_cumprod: A `torch.Tensor`. The cumprod alphas for the discrete-time DPM. (See the original DDPM paper for details)

            Note that we always have alphas_cumprod = cumprod(1 - betas). Therefore, we only need to set one of `betas` and `alphas_cumprod`.

            **Important**:  Please pay special attention for the args for `alphas_cumprod`:
                The `alphas_cumprod` is the \hat{alpha_n} arrays in the notations of DDPM. Specifically, DDPMs assume that
                    q_{t_n | 0}(x_{t_n} | x_0) = N ( \sqrt{\hat{alpha_n}} * x_0, (1 - \hat{alpha_n}) * I ).
                Therefore, the notation \hat{alpha_n} is different from the notation alpha_t in DPM-Solver. In fact, we have
                    alpha_{t_n} = \sqrt{\hat{alpha_n}},
                and
                    log(alpha_{t_n}) = 0.5 * log(\hat{alpha_n}).


        2. For continuous-time DPMs:

            We support two types of VPSDEs: linear (DDPM) and cosine (improved-DDPM). The hyperparameters for the noise
            schedule are the default settings in DDPM and improved-DDPM:

            Args:
                beta_min: A `float` number. The smallest beta for the linear schedule.
                beta_max: A `float` number. The largest beta for the linear schedule.
                cosine_s: A `float` number. The hyperparameter in the cosine schedule.
                cosine_beta_max: A `float` number. The hyperparameter in the cosine schedule.
                T: A `float` number. The ending time of the forward process.

        ===============================================================

        Args:
            schedule: A `str`. The noise schedule of the forward SDE. 'discrete' for discrete-time DPMs,
                    'linear' or 'cosine' for continuous-time DPMs.
        Returns:
            A wrapper object of the forward SDE (VP type).
        
        ===============================================================

        Example:

        # For discrete-time DPMs, given betas (the beta array for n = 0, 1, ..., N - 1):
        >>> ns = NoiseScheduleVP('discrete', betas=betas)

        # For discrete-time DPMs, given alphas_cumprod (the \hat{alpha_n} array for n = 0, 1, ..., N - 1):
        >>> ns = NoiseScheduleVP('discrete', alphas_cumprod=alphas_cumprod)

        # For continuous-time DPMs (VPSDE), linear schedule:
        >>> ns = NoiseScheduleVP('linear', continuous_beta_0=0.1, continuous_beta_1=20.)

        """

        if schedule not in ['discrete', 'linear', 'cosine']:
            raise ValueError("Unsupported noise schedule {}. The schedule needs to be 'discrete' or 'linear' or 'cosine'".format(schedule))

        self.schedule = schedule
        if schedule == 'discrete':
            if betas is not None:
                log_alphas = 0.5 * torch.log(1 - betas).cumsum(dim=0)
            else:
                assert alphas_cumprod is not None
                log_alphas = 0.5 * torch.log(alphas_cumprod)
            self.total_N = len(log_alphas)
            self.T = 1.
            self.t_array = torch.linspace(0., 1., self.total_N + 1)[1:].reshape((1, -1)).to(dtype=dtype)
            self.log_alpha_array = log_alphas.reshape((1, -1,)).to(dtype=dtype)
        else:
            self.total_N = 1000
            self.beta_0 = continuous_beta_0
            self.beta_1 = continuous_beta_1
            self.cosine_s = 0.008
            self.cosine_beta_max = 999.
            self.cosine_t_max = math.atan(self.cosine_beta_max * (1. + self.cosine_s) / math.pi) * 2. * (1. + self.cosine_s) / math.pi - self.cosine_s
            self.cosine_log_alpha_0 = math.log(math.cos(self.cosine_s / (1. + self.cosine_s) * math.pi / 2.))
            self.schedule = schedule
            if schedule == 'cosine':
                # For the cosine schedule, T = 1 will have numerical issues. So we manually set the ending time T.
                # Note that T = 0.9946 may be not the optimal setting. However, we find it works well.
                self.T = 0.9946
            else:
                self.T = 1.

    def marginal_log_mean_coeff(self, t):
        """
        Compute log(alpha_t) of a given continuous-time label t in [0, T].
        """
        if self.schedule == 'discrete':
            return interpolate_fn(t.reshape((-1, 1)), self.t_array.to(t.device), self.log_alpha_array.to(t.device)).reshape((-1))
        elif self.schedule == 'linear':
            return -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
        elif self.schedule == 'cosine':
            log_alpha_fn = lambda s: torch.log(torch.cos((s + self.cosine_s) / (1. + self.cosine_s) * math.pi / 2.))
            log_alpha_t =  log_alpha_fn(t) - self.cosine_log_alpha_0
            return log_alpha_t

    def marginal_alpha(self, t):
        """
        Compute alpha_t of a given continuous-time label t in [0, T].
        """
        return torch.exp(self.marginal_log_mean_coeff(t))

    def marginal_std(self, t):
        """
        Compute sigma_t of a given continuous-time label t in [0, T].
        """
        return torch.sqrt(1. - torch.exp(2. * self.marginal_log_mean_coeff(t)))

    def marginal_lambda(self, t):
        """
        Compute lambda_t = log(alpha_t) - log(sigma_t) of a given continuous-time label t in [0, T].
        """
        log_mean_coeff = self.marginal_log_mean_coeff(t)
        log_std = 0.5 * torch.log(1. - torch.exp(2. * log_mean_coeff))
        return log_mean_coeff - log_std

    def inverse_lambda(self, lamb):
        """
        Compute the continuous-time label t in [0, T] of a given half-logSNR lambda_t.
        """
        if self.schedule == 'linear':
            tmp = 2. * (self.beta_1 - self.beta_0) * torch.logaddexp(-2. * lamb, torch.zeros((1,)).to(lamb))
            Delta = self.beta_0**2 + tmp
            return tmp / (torch.sqrt(Delta) + self.beta_0) / (self.beta_1 - self.beta_0)
        elif self.schedule == 'discrete':
            log_alpha = -0.5 * torch.logaddexp(torch.zeros((1,)).to(lamb.device), -2. * lamb)
            t = interpolate_fn(log_alpha.reshape((-1, 1)), torch.flip(self.log_alpha_array.to(lamb.device), [1]), torch.flip(self.t_array.to(lamb.device), [1]))
            return t.reshape((-1,))
        else:
            log_alpha = -0.5 * torch.logaddexp(-2. * lamb, torch.zeros((1,)).to(lamb))
            t_fn = lambda log_alpha_t: torch.arccos(torch.exp(log_alpha_t + self.cosine_log_alpha_0)) * 2. * (1. + self.cosine_s) / math.pi - self.cosine_s
            t = t_fn(log_alpha)
            return t


def model_wrapper(
    model,
    noise_schedule,
    model_type="noise",
    # model_kwargs={},
    guidance_type="uncond",
    condition=None,
    unconditional_condition=None,
    guidance_scale=1.,
    classifier_fn=None,
    classifier_kwargs={},
):
    """Create a wrapper function for the noise prediction model.

    DPM-Solver needs to solve the continuous-time diffusion ODEs. For DPMs trained on discrete-time labels, we need to
    firstly wrap the model function to a noise prediction model that accepts the continuous time as the input.

    We support four types of the diffusion model by setting `model_type`:

        1. "noise": noise prediction model. (Trained by predicting noise).

        2. "x_start": data prediction model. (Trained by predicting the data x_0 at time 0).

        3. "v": velocity prediction model. (Trained by predicting the velocity).
            The "v" prediction is derivation detailed in Appendix D of [1], and is used in Imagen-Video [2].

            [1] Salimans, Tim, and Jonathan Ho. "Progressive distillation for fast sampling of diffusion models."
                arXiv preprint arXiv:2202.00512 (2022).
            [2] Ho, Jonathan, et al. "Imagen Video: High Definition Video Generation with Diffusion Models."
                arXiv preprint arXiv:2210.02303 (2022).
    
        4. "score": marginal score function. (Trained by denoising score matching).
            Note that the score function and the noise prediction model follows a simple relationship:
            ```
                noise(x_t, t) = -sigma_t * score(x_t, t)
            ```

    We support three types of guided sampling by DPMs by setting `guidance_type`:
        1. "uncond": unconditional sampling by DPMs.
            The input `model` has the following format:
            ``
                model(x, t_input, **model_kwargs) -> noise | x_start | v | score
            ``

        2. "classifier": classifier guidance sampling [3] by DPMs and another classifier.
            The input `model` has the following format:
            ``
                model(x, t_input, **model_kwargs) -> noise | x_start | v | score
            `` 

            The input `classifier_fn` has the following format:
            ``
                classifier_fn(x, t_input, cond, **classifier_kwargs) -> logits(x, t_input, cond)
            ``

            [3] P. Dhariwal and A. Q. Nichol, "Diffusion models beat GANs on image synthesis,"
                in Advances in Neural Information Processing Systems, vol. 34, 2021, pp. 8780-8794.

        3. "classifier-free": classifier-free guidance sampling by conditional DPMs.
            The input `model` has the following format:
            ``
                model(x, t_input, cond, **model_kwargs) -> noise | x_start | v | score
            `` 
            And if cond == `unconditional_condition`, the model output is the unconditional DPM output.

            [4] Ho, Jonathan, and Tim Salimans. "Classifier-free diffusion guidance."
                arXiv preprint arXiv:2207.12598 (2022).
        

    The `t_input` is the time label of the model, which may be discrete-time labels (i.e. 0 to 999)
    or continuous-time labels (i.e. epsilon to T).

    We wrap the model function to accept only `x` and `t_continuous` as inputs, and outputs the predicted noise:
    ``
        def model_fn(x, t_continuous) -> noise:
            t_input = get_model_input_time(t_continuous)
            return noise_pred(model, x, t_input, **model_kwargs)         
    ``
    where `t_continuous` is the continuous time labels (i.e. epsilon to T). And we use `model_fn` for DPM-Solver.

    ===============================================================

    Args:
        model: A diffusion model with the corresponding format described above.
        noise_schedule: A noise schedule object, such as NoiseScheduleVP.
        model_type: A `str`. The parameterization type of the diffusion model.
                    "noise" or "x_start" or "v" or "score".
        model_kwargs: A `dict`. A dict for the other inputs of the model function.
        guidance_type: A `str`. The type of the guidance for sampling.
                    "uncond" or "classifier" or "classifier-free".
        condition: A pytorch tensor. The condition for the guided sampling.
                    Only used for "classifier" or "classifier-free" guidance type.
        unconditional_condition: A pytorch tensor. The condition for the unconditional sampling.
                    Only used for "classifier-free" guidance type.
        guidance_scale: A `float`. The scale for the guided sampling.
        classifier_fn: A classifier function. Only used for the classifier guidance.
        classifier_kwargs: A `dict`. A dict for the other inputs of the classifier function.
    Returns:
        A noise prediction model that accepts the noised data and the continuous time as the inputs.
    """

    def get_model_input_time(t_continuous):
        """
        Convert the continuous-time `t_continuous` (in [epsilon, T]) to the model input time.
        For discrete-time DPMs, we convert `t_continuous` in [1 / N, 1] to `t_input` in [0, 1000 * (N - 1) / N].
        For continuous-time DPMs, we just use `t_continuous`.
        """
        if noise_schedule.schedule == 'discrete':
            return (t_continuous - 1. / noise_schedule.total_N) * 1000.
        else:
            return t_continuous

    def noise_pred_fn(x, t_continuous, cond=None):
        t_input = get_model_input_time(t_continuous)
        if cond is None:
            output = model.apply_model(x, t_input)
        else:
            output = model.apply_model(x, t_input, cond)
        if model_type == "noise":
            return output
        elif model_type == "x_start":
            alpha_t, sigma_t = noise_schedule.marginal_alpha(t_continuous), noise_schedule.marginal_std(t_continuous)
            return (x - alpha_t * output) / sigma_t
        elif model_type == "v":
            alpha_t, sigma_t = noise_schedule.marginal_alpha(t_continuous), noise_schedule.marginal_std(t_continuous)
            return alpha_t * output + sigma_t * x
        elif model_type == "score":
            sigma_t = noise_schedule.marginal_std(t_continuous)
            return -sigma_t * output

    def cond_grad_fn(x, t_input):
        """
        Compute the gradient of the classifier, i.e. nabla_{x} log p_t(cond | x_t).
        """
        with torch.enable_grad():
            x_in = x.detach().requires_grad_(True)
            log_prob = classifier_fn(x_in, t_input, condition, **classifier_kwargs)
            return torch.autograd.grad(log_prob.sum(), x_in)[0]

    def model_fn(x, t_continuous):
        """
        The noise predicition model function that is used for DPM-Solver.
        """
        if guidance_type == "uncond":
            return noise_pred_fn(x, t_continuous)
        elif guidance_type == "classifier":
            assert classifier_fn is not None
            t_input = get_model_input_time(t_continuous)
            cond_grad = cond_grad_fn(x, t_input)
            sigma_t = noise_schedule.marginal_std(t_continuous)
            noise = noise_pred_fn(x, t_continuous)
            return noise - guidance_scale * sigma_t * cond_grad
        elif guidance_type == "classifier-free":
            if guidance_scale == 1. or unconditional_condition is None:
                return noise_pred_fn(x, t_continuous, cond=condition)
            else:
                x_in = torch.cat([x] * 2)
                t_in = torch.cat([t_continuous] * 2)
                c_in = torch.cat([unconditional_condition, condition])
                noise_uncond, noise = noise_pred_fn(x_in, t_in, cond=c_in).chunk(2)
                return noise_uncond + guidance_scale * (noise - noise_uncond)

    assert model_type in ["noise", "x_start", "v", "score"]
    assert guidance_type in ["uncond", "classifier", "classifier-free"]
    return model_fn


class DPM_Solver:
    def __init__(
        self,
        model_fn,
        noise_schedule,
        algorithm_type="dpmsolver++",
        correcting_x0_fn=None,
        correcting_xt_fn=None,
        thresholding_max_val=1.,
        dynamic_thresholding_ratio=0.995,
    ):
        """Construct a DPM-Solver. 

        We support both DPM-Solver (`algorithm_type="dpmsolver"`) and DPM-Solver++ (`algorithm_type="dpmsolver++"`).

        We also support the "dynamic thresholding" method in Imagen[1]. For pixel-space diffusion models, you
        can set both `algorithm_type="dpmsolver++"` and `correcting_x0_fn="dynamic_thresholding"` to use the
        dynamic thresholding. The "dynamic thresholding" can greatly improve the sample quality for pixel-space
        DPMs with large guidance scales. Note that the thresholding method is **unsuitable** for latent-space
        DPMs (such as stable-diffusion).

        To support advanced algorithms in image-to-image applications, we also support corrector functions for
        both x0 and xt.

        Args:
            model_fn: A noise prediction model function which accepts the continuous-time input (t in [epsilon, T]):
                ``
                def model_fn(x, t_continuous):
                    return noise
                ``
                The shape of `x` is `(batch_size, **shape)`, and the shape of `t_continuous` is `(batch_size,)`.
            noise_schedule: A noise schedule object, such as NoiseScheduleVP.
            algorithm_type: A `str`. Either "dpmsolver" or "dpmsolver++".
            correcting_x0_fn: A `str` or a function with the following format:
                ```
                def correcting_x0_fn(x0, t):
                    x0_new = ...
                    return x0_new
                ```
                This function is to correct the outputs of the data prediction model at each sampling step. e.g.,
                ```
                x0_pred = data_pred_model(xt, t)
                if correcting_x0_fn is not None:
                    x0_pred = correcting_x0_fn(x0_pred, t)
                xt_1 = update(x0_pred, xt, t)
                ```
                If `correcting_x0_fn="dynamic_thresholding"`, we use the dynamic thresholding proposed in Imagen[1].
            correcting_xt_fn: A function with the following format:
                ```
                def correcting_xt_fn(xt, t, step):
                    x_new = ...
                    return x_new
                ```
                This function is to correct the intermediate samples xt at each sampling step. e.g.,
                ```
                xt = ...
                xt = correcting_xt_fn(xt, t, step)
                ```
            thresholding_max_val: A `float`. The max value for thresholding.
                Valid only when use `dpmsolver++` and `correcting_x0_fn="dynamic_thresholding"`.
            dynamic_thresholding_ratio: A `float`. The ratio for dynamic thresholding (see Imagen[1] for details).
                Valid only when use `dpmsolver++` and `correcting_x0_fn="dynamic_thresholding"`.

        [1] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily Denton, Seyed Kamyar Seyed Ghasemipour,
            Burcu Karagol Ayan, S Sara Mahdavi, Rapha Gontijo Lopes, et al. Photorealistic text-to-image diffusion models
            with deep language understanding. arXiv preprint arXiv:2205.11487, 2022b.
        """
        self.model = lambda x, t: model_fn(x, t.expand((x.shape[0])))
        self.noise_schedule = noise_schedule
        assert algorithm_type in ["dpmsolver", "dpmsolver++"]
        self.algorithm_type = algorithm_type
        if correcting_x0_fn == "dynamic_thresholding":
            self.correcting_x0_fn = self.dynamic_thresholding_fn
        else:
            self.correcting_x0_fn = correcting_x0_fn
        self.correcting_xt_fn = correcting_xt_fn
        self.dynamic_thresholding_ratio = dynamic_thresholding_ratio
        self.thresholding_max_val = thresholding_max_val

    def dynamic_thresholding_fn(self, x0, t):
        """
        The dynamic thresholding method. 
        """
        dims = x0.dim()
        p = self.dynamic_thresholding_ratio
        s = torch.quantile(torch.abs(x0).reshape((x0.shape[0], -1)), p, dim=1)
        s = expand_dims(torch.maximum(s, self.thresholding_max_val * torch.ones_like(s).to(s.device)), dims)
        x0 = torch.clamp(x0, -s, s) / s
        return x0

    def noise_prediction_fn(self, x, t):
        """
        Return the noise prediction model.
        """
        return self.model(x, t)

    def data_prediction_fn(self, x, t):
        """
        Return the data prediction model (with corrector).
        """
        noise = self.noise_prediction_fn(x, t)
        alpha_t, sigma_t = self.noise_schedule.marginal_alpha(t), self.noise_schedule.marginal_std(t)
        x0 = (x - sigma_t * noise) / alpha_t
        if self.correcting_x0_fn is not None:
            x0 = self.correcting_x0_fn(x0, t)
        return x0

    def model_fn(self, x, t):
        """
        Convert the model to the noise prediction model or the data prediction model. 
        """
        if self.algorithm_type == "dpmsolver++":
            return self.data_prediction_fn(x, t)
        else:
            return self.noise_prediction_fn(x, t)

    def get_time_steps(self, skip_type, t_T, t_0, N, device):
        """Compute the intermediate time steps for sampling.

        Args:
            skip_type: A `str`. The type for the spacing of the time steps. We support three types:
                - 'logSNR': uniform logSNR for the time steps.
                - 'time_uniform': uniform time for the time steps. (**Recommended for high-resolutional data**.)
                - 'time_quadratic': quadratic time for the time steps. (Used in DDIM for low-resolutional data.)
            t_T: A `float`. The starting time of the sampling (default is T).
            t_0: A `float`. The ending time of the sampling (default is epsilon).
            N: A `int`. The total number of the spacing of the time steps.
            device: A torch device.
        Returns:
            A pytorch tensor of the time steps, with the shape (N + 1,).
        """
        if skip_type == 'logSNR':
            lambda_T = self.noise_schedule.marginal_lambda(torch.tensor(t_T).to(device))
            lambda_0 = self.noise_schedule.marginal_lambda(torch.tensor(t_0).to(device))
            logSNR_steps = torch.linspace(lambda_T.cpu().item(), lambda_0.cpu().item(), N + 1).to(device)
            return self.noise_schedule.inverse_lambda(logSNR_steps)
        elif skip_type == 'time_uniform':
            return torch.linspace(t_T, t_0, N + 1).to(device)
        elif skip_type == 'time_quadratic':
            t_order = 2
            t = torch.linspace(t_T**(1. / t_order), t_0**(1. / t_order), N + 1).pow(t_order).to(device)
            return t
        else:
            raise ValueError("Unsupported skip_type {}, need to be 'logSNR' or 'time_uniform' or 'time_quadratic'".format(skip_type))

    def get_orders_and_timesteps_for_singlestep_solver(self, steps, order, skip_type, t_T, t_0, device):
        """
        Get the order of each step for sampling by the singlestep DPM-Solver.

        We combine both DPM-Solver-1,2,3 to use all the function evaluations, which is named as "DPM-Solver-fast".
        Given a fixed number of function evaluations by `steps`, the sampling procedure by DPM-Solver-fast is:
            - If order == 1:
                We take `steps` of DPM-Solver-1 (i.e. DDIM).
            - If order == 2:
                - Denote K = (steps // 2). We take K or (K + 1) intermediate time steps for sampling.
                - If steps % 2 == 0, we use K steps of DPM-Solver-2.
                - If steps % 2 == 1, we use K steps of DPM-Solver-2 and 1 step of DPM-Solver-1.
            - If order == 3:
                - Denote K = (steps // 3 + 1). We take K intermediate time steps for sampling.
                - If steps % 3 == 0, we use (K - 2) steps of DPM-Solver-3, and 1 step of DPM-Solver-2 and 1 step of DPM-Solver-1.
                - If steps % 3 == 1, we use (K - 1) steps of DPM-Solver-3 and 1 step of DPM-Solver-1.
                - If steps % 3 == 2, we use (K - 1) steps of DPM-Solver-3 and 1 step of DPM-Solver-2.

        ============================================
        Args:
            order: A `int`. The max order for the solver (2 or 3).
            steps: A `int`. The total number of function evaluations (NFE).
            skip_type: A `str`. The type for the spacing of the time steps. We support three types:
                - 'logSNR': uniform logSNR for the time steps.
                - 'time_uniform': uniform time for the time steps. (**Recommended for high-resolutional data**.)
                - 'time_quadratic': quadratic time for the time steps. (Used in DDIM for low-resolutional data.)
            t_T: A `float`. The starting time of the sampling (default is T).
            t_0: A `float`. The ending time of the sampling (default is epsilon).
            device: A torch device.
        Returns:
            orders: A list of the solver order of each step.
        """
        if order == 3:
            K = steps // 3 + 1
            if steps % 3 == 0:
                orders = [3,] * (K - 2) + [2, 1]
            elif steps % 3 == 1:
                orders = [3,] * (K - 1) + [1]
            else:
                orders = [3,] * (K - 1) + [2]
        elif order == 2:
            if steps % 2 == 0:
                K = steps // 2
                orders = [2,] * K
            else:
                K = steps // 2 + 1
                orders = [2,] * (K - 1) + [1]
        elif order == 1:
            K = 1
            orders = [1,] * steps
        else:
            raise ValueError("'order' must be '1' or '2' or '3'.")
        if skip_type == 'logSNR':
            # To reproduce the results in DPM-Solver paper
            timesteps_outer = self.get_time_steps(skip_type, t_T, t_0, K, device)
        else:
            timesteps_outer = self.get_time_steps(skip_type, t_T, t_0, steps, device)[torch.cumsum(torch.tensor([0,] + orders), 0).to(device)]
        return timesteps_outer, orders

    def denoise_to_zero_fn(self, x, s):
        """
        Denoise at the final step, which is equivalent to solve the ODE from lambda_s to infty by first-order discretization. 
        """
        return self.data_prediction_fn(x, s)

    def dpm_solver_first_update(self, x, s, t, model_s=None, return_intermediate=False):
        """
        DPM-Solver-1 (equivalent to DDIM) from time `s` to time `t`.

        Args:
            x: A pytorch tensor. The initial value at time `s`.
            s: A pytorch tensor. The starting time, with the shape (1,).
            t: A pytorch tensor. The ending time, with the shape (1,).
            model_s: A pytorch tensor. The model function evaluated at time `s`.
                If `model_s` is None, we evaluate the model by `x` and `s`; otherwise we directly use it.
            return_intermediate: A `bool`. If true, also return the model value at time `s`.
        Returns:
            x_t: A pytorch tensor. The approximated solution at time `t`.
        """
        ns = self.noise_schedule
        dims = x.dim()
        lambda_s, lambda_t = ns.marginal_lambda(s), ns.marginal_lambda(t)
        h = lambda_t - lambda_s
        log_alpha_s, log_alpha_t = ns.marginal_log_mean_coeff(s), ns.marginal_log_mean_coeff(t)
        sigma_s, sigma_t = ns.marginal_std(s), ns.marginal_std(t)
        alpha_t = torch.exp(log_alpha_t)

        if self.algorithm_type == "dpmsolver++":
            phi_1 = torch.expm1(-h)
            if model_s is None:
                model_s = self.model_fn(x, s)
            x_t = (
                sigma_t / sigma_s * x
                - alpha_t * phi_1 * model_s
            )
            if return_intermediate:
                return x_t, {'model_s': model_s}
            else:
                return x_t
        else:
            phi_1 = torch.expm1(h)
            if model_s is None:
                model_s = self.model_fn(x, s)
            x_t = (
                torch.exp(log_alpha_t - log_alpha_s) * x
                - (sigma_t * phi_1) * model_s
            )
            if return_intermediate:
                return x_t, {'model_s': model_s}
            else:
                return x_t

    def singlestep_dpm_solver_second_update(self, x, s, t, r1=0.5, model_s=None, return_intermediate=False, solver_type='dpmsolver'):
        """
        Singlestep solver DPM-Solver-2 from time `s` to time `t`.

        Args:
            x: A pytorch tensor. The initial value at time `s`.
            s: A pytorch tensor. The starting time, with the shape (1,).
            t: A pytorch tensor. The ending time, with the shape (1,).
            r1: A `float`. The hyperparameter of the second-order solver.
            model_s: A pytorch tensor. The model function evaluated at time `s`.
                If `model_s` is None, we evaluate the model by `x` and `s`; otherwise we directly use it.
            return_intermediate: A `bool`. If true, also return the model value at time `s` and `s1` (the intermediate time).
            solver_type: either 'dpmsolver' or 'taylor'. The type for the high-order solvers.
                The type slightly impacts the performance. We recommend to use 'dpmsolver' type.
        Returns:
            x_t: A pytorch tensor. The approximated solution at time `t`.
        """
        if solver_type not in ['dpmsolver', 'taylor']:
            raise ValueError("'solver_type' must be either 'dpmsolver' or 'taylor', got {}".format(solver_type))
        if r1 is None:
            r1 = 0.5
        ns = self.noise_schedule
        lambda_s, lambda_t = ns.marginal_lambda(s), ns.marginal_lambda(t)
        h = lambda_t - lambda_s
        lambda_s1 = lambda_s + r1 * h
        s1 = ns.inverse_lambda(lambda_s1)
        log_alpha_s, log_alpha_s1, log_alpha_t = ns.marginal_log_mean_coeff(s), ns.marginal_log_mean_coeff(s1), ns.marginal_log_mean_coeff(t)
        sigma_s, sigma_s1, sigma_t = ns.marginal_std(s), ns.marginal_std(s1), ns.marginal_std(t)
        alpha_s1, alpha_t = torch.exp(log_alpha_s1), torch.exp(log_alpha_t)

        if self.algorithm_type == "dpmsolver++":
            phi_11 = torch.expm1(-r1 * h)
            phi_1 = torch.expm1(-h)

            if model_s is None:
                model_s = self.model_fn(x, s)
            x_s1 = (
                (sigma_s1 / sigma_s) * x
                - (alpha_s1 * phi_11) * model_s
            )
            model_s1 = self.model_fn(x_s1, s1)
            if solver_type == 'dpmsolver':
                x_t = (
                    (sigma_t / sigma_s) * x
                    - (alpha_t * phi_1) * model_s
                    - (0.5 / r1) * (alpha_t * phi_1) * (model_s1 - model_s)
                )
            elif solver_type == 'taylor':
                x_t = (
                    (sigma_t / sigma_s) * x
                    - (alpha_t * phi_1) * model_s
                    + (1. / r1) * (alpha_t * (phi_1 / h + 1.)) * (model_s1 - model_s)
                )
        else:
            phi_11 = torch.expm1(r1 * h)
            phi_1 = torch.expm1(h)

            if model_s is None:
                model_s = self.model_fn(x, s)
            x_s1 = (
                torch.exp(log_alpha_s1 - log_alpha_s) * x
                - (sigma_s1 * phi_11) * model_s
            )
            model_s1 = self.model_fn(x_s1, s1)
            if solver_type == 'dpmsolver':
                x_t = (
                    torch.exp(log_alpha_t - log_alpha_s) * x
                    - (sigma_t * phi_1) * model_s
                    - (0.5 / r1) * (sigma_t * phi_1) * (model_s1 - model_s)
                )
            elif solver_type == 'taylor':
                x_t = (
                    torch.exp(log_alpha_t - log_alpha_s) * x
                    - (sigma_t * phi_1) * model_s
                    - (1. / r1) * (sigma_t * (phi_1 / h - 1.)) * (model_s1 - model_s)
                )
        if return_intermediate:
            return x_t, {'model_s': model_s, 'model_s1': model_s1, 'x_s1': x_s1, 's1': s1}
        else:
            return x_t

    def singlestep_dpm_solver_third_update(self, x, s, t, r1=1./3., r2=2./3., model_s=None, model_s1=None, return_intermediate=False, solver_type='dpmsolver'):
        """
        Singlestep solver DPM-Solver-3 from time `s` to time `t`.

        Args:
            x: A pytorch tensor. The initial value at time `s`.
            s: A pytorch tensor. The starting time, with the shape (1,).
            t: A pytorch tensor. The ending time, with the shape (1,).
            r1: A `float`. The hyperparameter of the third-order solver.
            r2: A `float`. The hyperparameter of the third-order solver.
            model_s: A pytorch tensor. The model function evaluated at time `s`.
                If `model_s` is None, we evaluate the model by `x` and `s`; otherwise we directly use it.
            model_s1: A pytorch tensor. The model function evaluated at time `s1` (the intermediate time given by `r1`).
                If `model_s1` is None, we evaluate the model at `s1`; otherwise we directly use it.
            return_intermediate: A `bool`. If true, also return the model value at time `s`, `s1` and `s2` (the intermediate times).
            solver_type: either 'dpmsolver' or 'taylor'. The type for the high-order solvers.
                The type slightly impacts the performance. We recommend to use 'dpmsolver' type.
        Returns:
            x_t: A pytorch tensor. The approximated solution at time `t`.
        """
        if solver_type not in ['dpmsolver', 'taylor']:
            raise ValueError("'solver_type' must be either 'dpmsolver' or 'taylor', got {}".format(solver_type))
        if r1 is None:
            r1 = 1. / 3.
        if r2 is None:
            r2 = 2. / 3.
        ns = self.noise_schedule
        lambda_s, lambda_t = ns.marginal_lambda(s), ns.marginal_lambda(t)
        h = lambda_t - lambda_s
        lambda_s1 = lambda_s + r1 * h
        lambda_s2 = lambda_s + r2 * h
        s1 = ns.inverse_lambda(lambda_s1)
        s2 = ns.inverse_lambda(lambda_s2)
        log_alpha_s, log_alpha_s1, log_alpha_s2, log_alpha_t = ns.marginal_log_mean_coeff(s), ns.marginal_log_mean_coeff(s1), ns.marginal_log_mean_coeff(s2), ns.marginal_log_mean_coeff(t)
        sigma_s, sigma_s1, sigma_s2, sigma_t = ns.marginal_std(s), ns.marginal_std(s1), ns.marginal_std(s2), ns.marginal_std(t)
        alpha_s1, alpha_s2, alpha_t = torch.exp(log_alpha_s1), torch.exp(log_alpha_s2), torch.exp(log_alpha_t)

        if self.algorithm_type == "dpmsolver++":
            phi_11 = torch.expm1(-r1 * h)
            phi_12 = torch.expm1(-r2 * h)
            phi_1 = torch.expm1(-h)
            phi_22 = torch.expm1(-r2 * h) / (r2 * h) + 1.
            phi_2 = phi_1 / h + 1.
            phi_3 = phi_2 / h - 0.5

            if model_s is None:
                model_s = self.model_fn(x, s)
            if model_s1 is None:
                x_s1 = (
                    (sigma_s1 / sigma_s) * x
                    - (alpha_s1 * phi_11) * model_s
                )
                model_s1 = self.model_fn(x_s1, s1)
            x_s2 = (
                (sigma_s2 / sigma_s) * x
                - (alpha_s2 * phi_12) * model_s
                + r2 / r1 * (alpha_s2 * phi_22) * (model_s1 - model_s)
            )
            model_s2 = self.model_fn(x_s2, s2)
            if solver_type == 'dpmsolver':
                x_t = (
                    (sigma_t / sigma_s) * x
                    - (alpha_t * phi_1) * model_s
                    + (1. / r2) * (alpha_t * phi_2) * (model_s2 - model_s)
                )
            elif solver_type == 'taylor':
                D1_0 = (1. / r1) * (model_s1 - model_s)
                D1_1 = (1. / r2) * (model_s2 - model_s)
                D1 = (r2 * D1_0 - r1 * D1_1) / (r2 - r1)
                D2 = 2. * (D1_1 - D1_0) / (r2 - r1)
                x_t = (
                    (sigma_t / sigma_s) * x
                    - (alpha_t * phi_1) * model_s
                    + (alpha_t * phi_2) * D1
                    - (alpha_t * phi_3) * D2
                )
        else:
            phi_11 = torch.expm1(r1 * h)
            phi_12 = torch.expm1(r2 * h)
            phi_1 = torch.expm1(h)
            phi_22 = torch.expm1(r2 * h) / (r2 * h) - 1.
            phi_2 = phi_1 / h - 1.
            phi_3 = phi_2 / h - 0.5

            if model_s is None:
                model_s = self.model_fn(x, s)
            if model_s1 is None:
                x_s1 = (
                    (torch.exp(log_alpha_s1 - log_alpha_s)) * x
                    - (sigma_s1 * phi_11) * model_s
                )
                model_s1 = self.model_fn(x_s1, s1)
            x_s2 = (
                (torch.exp(log_alpha_s2 - log_alpha_s)) * x
                - (sigma_s2 * phi_12) * model_s
                - r2 / r1 * (sigma_s2 * phi_22) * (model_s1 - model_s)
            )
            model_s2 = self.model_fn(x_s2, s2)
            if solver_type == 'dpmsolver':
                x_t = (
                    (torch.exp(log_alpha_t - log_alpha_s)) * x
                    - (sigma_t * phi_1) * model_s
                    - (1. / r2) * (sigma_t * phi_2) * (model_s2 - model_s)
                )
            elif solver_type == 'taylor':
                D1_0 = (1. / r1) * (model_s1 - model_s)
                D1_1 = (1. / r2) * (model_s2 - model_s)
                D1 = (r2 * D1_0 - r1 * D1_1) / (r2 - r1)
                D2 = 2. * (D1_1 - D1_0) / (r2 - r1)
                x_t = (
                    (torch.exp(log_alpha_t - log_alpha_s)) * x
                    - (sigma_t * phi_1) * model_s
                    - (sigma_t * phi_2) * D1
                    - (sigma_t * phi_3) * D2
                )

        if return_intermediate:
            return x_t, {'model_s': model_s, 'model_s1': model_s1, 'model_s2': model_s2}
        else:
            return x_t

    def multistep_dpm_solver_second_update(self, x, model_prev_list, t_prev_list, t, solver_type="dpmsolver"):
        """
        Multistep solver DPM-Solver-2 from time `t_prev_list[-1]` to time `t`.

        Args:
            x: A pytorch tensor. The initial value at time `s`.
            model_prev_list: A list of pytorch tensor. The previous computed model values.
            t_prev_list: A list of pytorch tensor. The previous times, each time has the shape (1,)
            t: A pytorch tensor. The ending time, with the shape (1,).
            solver_type: either 'dpmsolver' or 'taylor'. The type for the high-order solvers.
                The type slightly impacts the performance. We recommend to use 'dpmsolver' type.
        Returns:
            x_t: A pytorch tensor. The approximated solution at time `t`.
        """
        if solver_type not in ['dpmsolver', 'taylor']:
            raise ValueError("'solver_type' must be either 'dpmsolver' or 'taylor', got {}".format(solver_type))
        ns = self.noise_schedule
        model_prev_1, model_prev_0 = model_prev_list[-2], model_prev_list[-1]
        t_prev_1, t_prev_0 = t_prev_list[-2], t_prev_list[-1]
        lambda_prev_1, lambda_prev_0, lambda_t = ns.marginal_lambda(t_prev_1), ns.marginal_lambda(t_prev_0), ns.marginal_lambda(t)
        log_alpha_prev_0, log_alpha_t = ns.marginal_log_mean_coeff(t_prev_0), ns.marginal_log_mean_coeff(t)
        sigma_prev_0, sigma_t = ns.marginal_std(t_prev_0), ns.marginal_std(t)
        alpha_t = torch.exp(log_alpha_t)

        h_0 = lambda_prev_0 - lambda_prev_1
        h = lambda_t - lambda_prev_0
        r0 = h_0 / h
        D1_0 = (1. / r0) * (model_prev_0 - model_prev_1)
        if self.algorithm_type == "dpmsolver++":
            phi_1 = torch.expm1(-h)
            if solver_type == 'dpmsolver':
                x_t = (
                    (sigma_t / sigma_prev_0) * x
                    - (alpha_t * phi_1) * model_prev_0
                    - 0.5 * (alpha_t * phi_1) * D1_0
                )
            elif solver_type == 'taylor':
                x_t = (
                    (sigma_t / sigma_prev_0) * x
                    - (alpha_t * phi_1) * model_prev_0
                    + (alpha_t * (phi_1 / h + 1.)) * D1_0
                )
        else:
            phi_1 = torch.expm1(h)
            if solver_type == 'dpmsolver':
                x_t = (
                    (torch.exp(log_alpha_t - log_alpha_prev_0)) * x
                    - (sigma_t * phi_1) * model_prev_0
                    - 0.5 * (sigma_t * phi_1) * D1_0
                )
            elif solver_type == 'taylor':
                x_t = (
                    (torch.exp(log_alpha_t - log_alpha_prev_0)) * x
                    - (sigma_t * phi_1) * model_prev_0
                    - (sigma_t * (phi_1 / h - 1.)) * D1_0
                )
        return x_t

    def multistep_dpm_solver_third_update(self, x, model_prev_list, t_prev_list, t, solver_type='dpmsolver'):
        """
        Multistep solver DPM-Solver-3 from time `t_prev_list[-1]` to time `t`.

        Args:
            x: A pytorch tensor. The initial value at time `s`.
            model_prev_list: A list of pytorch tensor. The previous computed model values.
            t_prev_list: A list of pytorch tensor. The previous times, each time has the shape (1,)
            t: A pytorch tensor. The ending time, with the shape (1,).
            solver_type: either 'dpmsolver' or 'taylor'. The type for the high-order solvers.
                The type slightly impacts the performance. We recommend to use 'dpmsolver' type.
        Returns:
            x_t: A pytorch tensor. The approximated solution at time `t`.
        """
        ns = self.noise_schedule
        model_prev_2, model_prev_1, model_prev_0 = model_prev_list
        t_prev_2, t_prev_1, t_prev_0 = t_prev_list
        lambda_prev_2, lambda_prev_1, lambda_prev_0, lambda_t = ns.marginal_lambda(t_prev_2), ns.marginal_lambda(t_prev_1), ns.marginal_lambda(t_prev_0), ns.marginal_lambda(t)
        log_alpha_prev_0, log_alpha_t = ns.marginal_log_mean_coeff(t_prev_0), ns.marginal_log_mean_coeff(t)
        sigma_prev_0, sigma_t = ns.marginal_std(t_prev_0), ns.marginal_std(t)
        alpha_t = torch.exp(log_alpha_t)

        h_1 = lambda_prev_1 - lambda_prev_2
        h_0 = lambda_prev_0 - lambda_prev_1
        h = lambda_t - lambda_prev_0
        r0, r1 = h_0 / h, h_1 / h
        D1_0 = (1. / r0) * (model_prev_0 - model_prev_1)
        D1_1 = (1. / r1) * (model_prev_1 - model_prev_2)
        D1 = D1_0 + (r0 / (r0 + r1)) * (D1_0 - D1_1)
        D2 = (1. / (r0 + r1)) * (D1_0 - D1_1)
        if self.algorithm_type == "dpmsolver++":
            phi_1 = torch.expm1(-h)
            phi_2 = phi_1 / h + 1.
            phi_3 = phi_2 / h - 0.5
            x_t = (
                (sigma_t / sigma_prev_0) * x
                - (alpha_t * phi_1) * model_prev_0
                + (alpha_t * phi_2) * D1
                - (alpha_t * phi_3) * D2
            )
        else:
            phi_1 = torch.expm1(h)
            phi_2 = phi_1 / h - 1.
            phi_3 = phi_2 / h - 0.5
            x_t = (
                (torch.exp(log_alpha_t - log_alpha_prev_0)) * x
                - (sigma_t * phi_1) * model_prev_0
                - (sigma_t * phi_2) * D1
                - (sigma_t * phi_3) * D2
            )
        return x_t

    def singlestep_dpm_solver_update(self, x, s, t, order, return_intermediate=False, solver_type='dpmsolver', r1=None, r2=None):
        """
        Singlestep DPM-Solver with the order `order` from time `s` to time `t`.

        Args:
            x: A pytorch tensor. The initial value at time `s`.
            s: A pytorch tensor. The starting time, with the shape (1,).
            t: A pytorch tensor. The ending time, with the shape (1,).
            order: A `int`. The order of DPM-Solver. We only support order == 1 or 2 or 3.
            return_intermediate: A `bool`. If true, also return the model value at time `s`, `s1` and `s2` (the intermediate times).
            solver_type: either 'dpmsolver' or 'taylor'. The type for the high-order solvers.
                The type slightly impacts the performance. We recommend to use 'dpmsolver' type.
            r1: A `float`. The hyperparameter of the second-order or third-order solver.
            r2: A `float`. The hyperparameter of the third-order solver.
        Returns:
            x_t: A pytorch tensor. The approximated solution at time `t`.
        """
        if order == 1:
            return self.dpm_solver_first_update(x, s, t, return_intermediate=return_intermediate)
        elif order == 2:
            return self.singlestep_dpm_solver_second_update(x, s, t, return_intermediate=return_intermediate, solver_type=solver_type, r1=r1)
        elif order == 3:
            return self.singlestep_dpm_solver_third_update(x, s, t, return_intermediate=return_intermediate, solver_type=solver_type, r1=r1, r2=r2)
        else:
            raise ValueError("Solver order must be 1 or 2 or 3, got {}".format(order))

    def multistep_dpm_solver_update(self, x, model_prev_list, t_prev_list, t, order, solver_type='dpmsolver'):
        """
        Multistep DPM-Solver with the order `order` from time `t_prev_list[-1]` to time `t`.

        Args:
            x: A pytorch tensor. The initial value at time `s`.
            model_prev_list: A list of pytorch tensor. The previous computed model values.
            t_prev_list: A list of pytorch tensor. The previous times, each time has the shape (1,)
            t: A pytorch tensor. The ending time, with the shape (1,).
            order: A `int`. The order of DPM-Solver. We only support order == 1 or 2 or 3.
            solver_type: either 'dpmsolver' or 'taylor'. The type for the high-order solvers.
                The type slightly impacts the performance. We recommend to use 'dpmsolver' type.
        Returns:
            x_t: A pytorch tensor. The approximated solution at time `t`.
        """
        if order == 1:
            return self.dpm_solver_first_update(x, t_prev_list[-1], t, model_s=model_prev_list[-1])
        elif order == 2:
            return self.multistep_dpm_solver_second_update(x, model_prev_list, t_prev_list, t, solver_type=solver_type)
        elif order == 3:
            return self.multistep_dpm_solver_third_update(x, model_prev_list, t_prev_list, t, solver_type=solver_type)
        else:
            raise ValueError("Solver order must be 1 or 2 or 3, got {}".format(order))

    def dpm_solver_adaptive(self, x, order, t_T, t_0, h_init=0.05, atol=0.0078, rtol=0.05, theta=0.9, t_err=1e-5, solver_type='dpmsolver'):
        """
        The adaptive step size solver based on singlestep DPM-Solver.

        Args:
            x: A pytorch tensor. The initial value at time `t_T`.
            order: A `int`. The (higher) order of the solver. We only support order == 2 or 3.
            t_T: A `float`. The starting time of the sampling (default is T).
            t_0: A `float`. The ending time of the sampling (default is epsilon).
            h_init: A `float`. The initial step size (for logSNR).
            atol: A `float`. The absolute tolerance of the solver. For image data, the default setting is 0.0078, followed [1].
            rtol: A `float`. The relative tolerance of the solver. The default setting is 0.05.
            theta: A `float`. The safety hyperparameter for adapting the step size. The default setting is 0.9, followed [1].
            t_err: A `float`. The tolerance for the time. We solve the diffusion ODE until the absolute error between the 
                current time and `t_0` is less than `t_err`. The default setting is 1e-5.
            solver_type: either 'dpmsolver' or 'taylor'. The type for the high-order solvers.
                The type slightly impacts the performance. We recommend to use 'dpmsolver' type.
        Returns:
            x_0: A pytorch tensor. The approximated solution at time `t_0`.

        [1] A. Jolicoeur-Martineau, K. Li, R. Piché-Taillefer, T. Kachman, and I. Mitliagkas, "Gotta go fast when generating data with score-based models," arXiv preprint arXiv:2105.14080, 2021.
        """
        ns = self.noise_schedule
        s = t_T * torch.ones((1,)).to(x)
        lambda_s = ns.marginal_lambda(s)
        lambda_0 = ns.marginal_lambda(t_0 * torch.ones_like(s).to(x))
        h = h_init * torch.ones_like(s).to(x)
        x_prev = x
        nfe = 0
        if order == 2:
            r1 = 0.5
            lower_update = lambda x, s, t: self.dpm_solver_first_update(x, s, t, return_intermediate=True)
            higher_update = lambda x, s, t, **kwargs: self.singlestep_dpm_solver_second_update(x, s, t, r1=r1, solver_type=solver_type, **kwargs)
        elif order == 3:
            r1, r2 = 1. / 3., 2. / 3.
            lower_update = lambda x, s, t: self.singlestep_dpm_solver_second_update(x, s, t, r1=r1, return_intermediate=True, solver_type=solver_type)
            higher_update = lambda x, s, t, **kwargs: self.singlestep_dpm_solver_third_update(x, s, t, r1=r1, r2=r2, solver_type=solver_type, **kwargs)
        else:
            raise ValueError("For adaptive step size solver, order must be 2 or 3, got {}".format(order))
        while torch.abs((s - t_0)).mean() > t_err:
            t = ns.inverse_lambda(lambda_s + h)
            x_lower, lower_noise_kwargs = lower_update(x, s, t)
            x_higher = higher_update(x, s, t, **lower_noise_kwargs)
            delta = torch.max(torch.ones_like(x).to(x) * atol, rtol * torch.max(torch.abs(x_lower), torch.abs(x_prev)))
            norm_fn = lambda v: torch.sqrt(torch.square(v.reshape((v.shape[0], -1))).mean(dim=-1, keepdim=True))
            E = norm_fn((x_higher - x_lower) / delta).max()
            if torch.all(E <= 1.):
                x = x_higher
                s = t
                x_prev = x_lower
                lambda_s = ns.marginal_lambda(s)
            h = torch.min(theta * h * torch.float_power(E, -1. / order).float(), lambda_0 - lambda_s)
            nfe += order
        print('adaptive solver nfe', nfe)
        return x

    def add_noise(self, x, t, noise=None):
        """
        Compute the noised input xt = alpha_t * x + sigma_t * noise. 

        Args:
            x: A `torch.Tensor` with shape `(batch_size, *shape)`.
            t: A `torch.Tensor` with shape `(t_size,)`.
        Returns:
            xt with shape `(t_size, batch_size, *shape)`.
        """
        alpha_t, sigma_t = self.noise_schedule.marginal_alpha(t), self.noise_schedule.marginal_std(t)
        if noise is None:
            noise = torch.randn((t.shape[0], *x.shape), device=x.device)
        x = x.reshape((-1, *x.shape))
        xt = expand_dims(alpha_t, x.dim()) * x + expand_dims(sigma_t, x.dim()) * noise
        if t.shape[0] == 1:
            return xt.squeeze(0)
        else:
            return xt

    def inverse(self, x, steps=20, t_start=None, t_end=None, order=2, skip_type='time_uniform',
        method='multistep', lower_order_final=True, denoise_to_zero=False, solver_type='dpmsolver',
        atol=0.0078, rtol=0.05, return_intermediate=False,
    ):
        """
        Inverse the sample `x` from time `t_start` to `t_end` by DPM-Solver.
        For discrete-time DPMs, we use `t_start=1/N`, where `N` is the total time steps during training.
        """
        t_0 = 1. / self.noise_schedule.total_N if t_start is None else t_start
        t_T = self.noise_schedule.T if t_end is None else t_end
        assert t_0 > 0 and t_T > 0, "Time range needs to be greater than 0. For discrete-time DPMs, it needs to be in [1 / N, 1], where N is the length of betas array"
        return self.sample(x, steps=steps, t_start=t_0, t_end=t_T, order=order, skip_type=skip_type,
            method=method, lower_order_final=lower_order_final, denoise_to_zero=denoise_to_zero, solver_type=solver_type,
            atol=atol, rtol=rtol, return_intermediate=return_intermediate)

    def sample(self, x, steps=20, t_start=None, t_end=None, order=2, skip_type='time_uniform',
        method='multistep', lower_order_final=True, denoise_to_zero=False, solver_type='dpmsolver',
        atol=0.0078, rtol=0.05, return_intermediate=False,
    ):
        """
        Compute the sample at time `t_end` by DPM-Solver, given the initial `x` at time `t_start`.

        =====================================================

        We support the following algorithms for both noise prediction model and data prediction model:
            - 'singlestep':
                Singlestep DPM-Solver (i.e. "DPM-Solver-fast" in the paper), which combines different orders of singlestep DPM-Solver. 
                We combine all the singlestep solvers with order <= `order` to use up all the function evaluations (steps).
                The total number of function evaluations (NFE) == `steps`.
                Given a fixed NFE == `steps`, the sampling procedure is:
                    - If `order` == 1:
                        - Denote K = steps. We use K steps of DPM-Solver-1 (i.e. DDIM).
                    - If `order` == 2:
                        - Denote K = (steps // 2) + (steps % 2). We take K intermediate time steps for sampling.
                        - If steps % 2 == 0, we use K steps of singlestep DPM-Solver-2.
                        - If steps % 2 == 1, we use (K - 1) steps of singlestep DPM-Solver-2 and 1 step of DPM-Solver-1.
                    - If `order` == 3:
                        - Denote K = (steps // 3 + 1). We take K intermediate time steps for sampling.
                        - If steps % 3 == 0, we use (K - 2) steps of singlestep DPM-Solver-3, and 1 step of singlestep DPM-Solver-2 and 1 step of DPM-Solver-1.
                        - If steps % 3 == 1, we use (K - 1) steps of singlestep DPM-Solver-3 and 1 step of DPM-Solver-1.
                        - If steps % 3 == 2, we use (K - 1) steps of singlestep DPM-Solver-3 and 1 step of singlestep DPM-Solver-2.
            - 'multistep':
                Multistep DPM-Solver with the order of `order`. The total number of function evaluations (NFE) == `steps`.
                We initialize the first `order` values by lower order multistep solvers.
                Given a fixed NFE == `steps`, the sampling procedure is:
                    Denote K = steps.
                    - If `order` == 1:
                        - We use K steps of DPM-Solver-1 (i.e. DDIM).
                    - If `order` == 2:
                        - We firstly use 1 step of DPM-Solver-1, then use (K - 1) step of multistep DPM-Solver-2.
                    - If `order` == 3:
                        - We firstly use 1 step of DPM-Solver-1, then 1 step of multistep DPM-Solver-2, then (K - 2) step of multistep DPM-Solver-3.
            - 'singlestep_fixed':
                Fixed order singlestep DPM-Solver (i.e. DPM-Solver-1 or singlestep DPM-Solver-2 or singlestep DPM-Solver-3).
                We use singlestep DPM-Solver-`order` for `order`=1 or 2 or 3, with total [`steps` // `order`] * `order` NFE.
            - 'adaptive':
                Adaptive step size DPM-Solver (i.e. "DPM-Solver-12" and "DPM-Solver-23" in the paper).
                We ignore `steps` and use adaptive step size DPM-Solver with a higher order of `order`.
                You can adjust the absolute tolerance `atol` and the relative tolerance `rtol` to balance the computatation costs
                (NFE) and the sample quality.
                    - If `order` == 2, we use DPM-Solver-12 which combines DPM-Solver-1 and singlestep DPM-Solver-2.
                    - If `order` == 3, we use DPM-Solver-23 which combines singlestep DPM-Solver-2 and singlestep DPM-Solver-3.

        =====================================================

        Some advices for choosing the algorithm:
            - For **unconditional sampling** or **guided sampling with small guidance scale** by DPMs:
                Use singlestep DPM-Solver or DPM-Solver++ ("DPM-Solver-fast" in the paper) with `order = 3`.
                e.g., DPM-Solver:
                    >>> dpm_solver = DPM_Solver(model_fn, noise_schedule, algorithm_type="dpmsolver")
                    >>> x_sample = dpm_solver.sample(x, steps=steps, t_start=t_start, t_end=t_end, order=3,
                            skip_type='time_uniform', method='singlestep')
                e.g., DPM-Solver++:
                    >>> dpm_solver = DPM_Solver(model_fn, noise_schedule, algorithm_type="dpmsolver++")
                    >>> x_sample = dpm_solver.sample(x, steps=steps, t_start=t_start, t_end=t_end, order=3,
                            skip_type='time_uniform', method='singlestep')
            - For **guided sampling with large guidance scale** by DPMs:
                Use multistep DPM-Solver with `algorithm_type="dpmsolver++"` and `order = 2`.
                e.g.
                    >>> dpm_solver = DPM_Solver(model_fn, noise_schedule, algorithm_type="dpmsolver++")
                    >>> x_sample = dpm_solver.sample(x, steps=steps, t_start=t_start, t_end=t_end, order=2,
                            skip_type='time_uniform', method='multistep')

        We support three types of `skip_type`:
            - 'logSNR': uniform logSNR for the time steps. **Recommended for low-resolutional images**
            - 'time_uniform': uniform time for the time steps. **Recommended for high-resolutional images**.
            - 'time_quadratic': quadratic time for the time steps.

        =====================================================
        Args:
            x: A pytorch tensor. The initial value at time `t_start`
                e.g. if `t_start` == T, then `x` is a sample from the standard normal distribution.
            steps: A `int`. The total number of function evaluations (NFE).
            t_start: A `float`. The starting time of the sampling.
                If `T` is None, we use self.noise_schedule.T (default is 1.0).
            t_end: A `float`. The ending time of the sampling.
                If `t_end` is None, we use 1. / self.noise_schedule.total_N.
                e.g. if total_N == 1000, we have `t_end` == 1e-3.
                For discrete-time DPMs:
                    - We recommend `t_end` == 1. / self.noise_schedule.total_N.
                For continuous-time DPMs:
                    - We recommend `t_end` == 1e-3 when `steps` <= 15; and `t_end` == 1e-4 when `steps` > 15.
            order: A `int`. The order of DPM-Solver.
            skip_type: A `str`. The type for the spacing of the time steps. 'time_uniform' or 'logSNR' or 'time_quadratic'.
            method: A `str`. The method for sampling. 'singlestep' or 'multistep' or 'singlestep_fixed' or 'adaptive'.
            denoise_to_zero: A `bool`. Whether to denoise to time 0 at the final step.
                Default is `False`. If `denoise_to_zero` is `True`, the total NFE is (`steps` + 1).

                This trick is firstly proposed by DDPM (https://arxiv.org/abs/2006.11239) and
                score_sde (https://arxiv.org/abs/2011.13456). Such trick can improve the FID
                for diffusion models sampling by diffusion SDEs for low-resolutional images
                (such as CIFAR-10). However, we observed that such trick does not matter for
                high-resolutional images. As it needs an additional NFE, we do not recommend
                it for high-resolutional images.
            lower_order_final: A `bool`. Whether to use lower order solvers at the final steps.
                Only valid for `method=multistep` and `steps < 15`. We empirically find that
                this trick is a key to stabilizing the sampling by DPM-Solver with very few steps
                (especially for steps <= 10). So we recommend to set it to be `True`.
            solver_type: A `str`. The taylor expansion type for the solver. `dpmsolver` or `taylor`. We recommend `dpmsolver`.
            atol: A `float`. The absolute tolerance of the adaptive step size solver. Valid when `method` == 'adaptive'.
            rtol: A `float`. The relative tolerance of the adaptive step size solver. Valid when `method` == 'adaptive'.
            return_intermediate: A `bool`. Whether to save the xt at each step.
                When set to `True`, method returns a tuple (x0, intermediates); when set to False, method returns only x0.
        Returns:
            x_end: A pytorch tensor. The approximated solution at time `t_end`.

        """
        t_0 = 1. / self.noise_schedule.total_N if t_end is None else t_end
        t_T = self.noise_schedule.T if t_start is None else t_start
        assert t_0 > 0 and t_T > 0, "Time range needs to be greater than 0. For discrete-time DPMs, it needs to be in [1 / N, 1], where N is the length of betas array"
        if return_intermediate:
            assert method in ['multistep', 'singlestep', 'singlestep_fixed'], "Cannot use adaptive solver when saving intermediate values"
        if self.correcting_xt_fn is not None:
            assert method in ['multistep', 'singlestep', 'singlestep_fixed'], "Cannot use adaptive solver when correcting_xt_fn is not None"
        device = x.device
        intermediates = []
        x_interms = []
        t_interms = []
        if order == 1:
            x_interms.append(x)

        with torch.no_grad():
            if method == 'adaptive':
                x = self.dpm_solver_adaptive(x, order=order, t_T=t_T, t_0=t_0, atol=atol, rtol=rtol, solver_type=solver_type)
            elif method == 'multistep':
                assert steps >= order
                timesteps = self.get_time_steps(skip_type=skip_type, t_T=t_T, t_0=t_0, N=steps, device=device)
                assert timesteps.shape[0] - 1 == steps
                # Init the initial values.
                step = 0
                t = timesteps[step]
                t_prev_list = [t]
                model_prev_list = [self.model_fn(x, t)]
                if self.correcting_xt_fn is not None:
                    x = self.correcting_xt_fn(x, t, step)
                if return_intermediate:
                    intermediates.append(x)
                # Init the first `order` values by lower order multistep DPM-Solver.
                for step in range(1, order):
                    t = timesteps[step]
                    x = self.multistep_dpm_solver_update(x, model_prev_list, t_prev_list, t, step, solver_type=solver_type)
                    if self.correcting_xt_fn is not None:
                        x = self.correcting_xt_fn(x, t, step)
                    if return_intermediate:
                        intermediates.append(x)
                    t_prev_list.append(t)
                    model_prev_list.append(self.model_fn(x, t))
                # Compute the remaining values by `order`-th order multistep DPM-Solver.
                for step in range(order, steps + 1):
                    t = timesteps[step]
                    # We only use lower order for steps < 10
                    if lower_order_final and steps < 10:
                        step_order = min(order, steps + 1 - step)
                    else:
                        step_order = order
                    x = self.multistep_dpm_solver_update(x, model_prev_list, t_prev_list, t, step_order, solver_type=solver_type)
                    if self.correcting_xt_fn is not None:
                        x = self.correcting_xt_fn(x, t, step)
                    if return_intermediate:
                        intermediates.append(x)
                    for i in range(order - 1):
                        t_prev_list[i] = t_prev_list[i + 1]
                        model_prev_list[i] = model_prev_list[i + 1]
                    t_prev_list[-1] = t
                    # We do not need to evaluate the final model value.
                    if step < steps:
                        model_prev_list[-1] = self.model_fn(x, t)
            elif method in ['singlestep', 'singlestep_fixed']:
                if method == 'singlestep':
                    timesteps_outer, orders = self.get_orders_and_timesteps_for_singlestep_solver(steps=steps, order=order, skip_type=skip_type, t_T=t_T, t_0=t_0, device=device)
                elif method == 'singlestep_fixed':
                    K = steps // order
                    orders = [order,] * K
                    timesteps_outer = self.get_time_steps(skip_type=skip_type, t_T=t_T, t_0=t_0, N=K, device=device)
                for step, order in enumerate(orders):
                    s, t = timesteps_outer[step], timesteps_outer[step + 1]
                    timesteps_inner = self.get_time_steps(skip_type=skip_type, t_T=s.item(), t_0=t.item(), N=order, device=device)
                    lambda_inner = self.noise_schedule.marginal_lambda(timesteps_inner)
                    h = lambda_inner[-1] - lambda_inner[0]
                    r1 = None if order <= 1 else (lambda_inner[1] - lambda_inner[0]) / h
                    r2 = None if order <= 2 else (lambda_inner[2] - lambda_inner[0]) / h
                    x, inter_dict = self.singlestep_dpm_solver_update(x, s, t, order, solver_type=solver_type, r1=r1, r2=r2, return_intermediate=return_intermediate)
                    if self.correcting_xt_fn is not None:
                        x = self.correcting_xt_fn(x, t, step)
                    if return_intermediate:
                        intermediates.append(x)

                        if order == 1:
                            x_interms.append(x)
                            t_interms.append(s)
                        elif order == 2:
                            x_interms.append(inter_dict['x_s1'])
                            t_interms.append(inter_dict['s1'])
            else:
                raise ValueError("Got wrong method {}".format(method))
            if denoise_to_zero:
                t = torch.ones((1,)).to(device) * t_0
                x = self.denoise_to_zero_fn(x, t)
                if self.correcting_xt_fn is not None:
                    x = self.correcting_xt_fn(x, t, step + 1)
                if return_intermediate:
                    intermediates.append(x)
        if return_intermediate:
            return x, intermediates, x_interms, t_interms
        else:
            return x, timesteps_outer



#############################################################
# other utility functions
#############################################################

def interpolate_fn(x, xp, yp):
    """
    A piecewise linear function y = f(x), using xp and yp as keypoints.
    We implement f(x) in a differentiable way (i.e. applicable for autograd).
    The function f(x) is well-defined for all x-axis. (For x beyond the bounds of xp, we use the outmost points of xp to define the linear function.)

    Args:
        x: PyTorch tensor with shape [N, C], where N is the batch size, C is the number of channels (we use C = 1 for DPM-Solver).
        xp: PyTorch tensor with shape [C, K], where K is the number of keypoints.
        yp: PyTorch tensor with shape [C, K].
    Returns:
        The function values f(x), with shape [N, C].
    """
    N, K = x.shape[0], xp.shape[1]
    all_x = torch.cat([x.unsqueeze(2), xp.unsqueeze(0).repeat((N, 1, 1))], dim=2)
    sorted_all_x, x_indices = torch.sort(all_x, dim=2)
    x_idx = torch.argmin(x_indices, dim=2)
    cand_start_idx = x_idx - 1
    start_idx = torch.where(
        torch.eq(x_idx, 0),
        torch.tensor(1, device=x.device),
        torch.where(
            torch.eq(x_idx, K), torch.tensor(K - 2, device=x.device), cand_start_idx,
        ),
    )
    end_idx = torch.where(torch.eq(start_idx, cand_start_idx), start_idx + 2, start_idx + 1)
    start_x = torch.gather(sorted_all_x, dim=2, index=start_idx.unsqueeze(2)).squeeze(2)
    end_x = torch.gather(sorted_all_x, dim=2, index=end_idx.unsqueeze(2)).squeeze(2)
    start_idx2 = torch.where(
        torch.eq(x_idx, 0),
        torch.tensor(0, device=x.device),
        torch.where(
            torch.eq(x_idx, K), torch.tensor(K - 2, device=x.device), cand_start_idx,
        ),
    )
    y_positions_expanded = yp.unsqueeze(0).expand(N, -1, -1)
    start_y = torch.gather(y_positions_expanded, dim=2, index=start_idx2.unsqueeze(2)).squeeze(2)
    end_y = torch.gather(y_positions_expanded, dim=2, index=(start_idx2 + 1).unsqueeze(2)).squeeze(2)
    cand = start_y + (x - start_x) * (end_y - start_y) / (end_x - start_x)
    return cand

def expand_dims(v, dims):
    """
    Expand the tensor `v` to the dim `dims`.

    Args:
        `v`: a PyTorch tensor with shape [N].
        `dim`: a `int`.
    Returns:
        a PyTorch tensor with shape [N, 1, 1, ..., 1] and the total dimension is `dims`.
    """
    return v[(...,) + (None,)*(dims - 1)]

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from functools import partial

from ldm.modules.diffusionmodules.util import make_ddim_sampling_parameters, make_ddim_timesteps, noise_like


# class DpmSolverSampler(object):
#     def __init__(self, model, schedule="linear", algorithm_type='dpmsolver', device=torch.device('cuda:0'), **kwargs):
#         super().__init__()
#         self.model = model
#         self.ddpm_num_timesteps = model.num_timesteps
#         self.schedule = schedule
#         self.algorithm_type = algorithm_type 
#         self.device = device

#     def register_buffer(self, name, attr):
#         if type(attr) == torch.Tensor:
#             if attr.device != self.device:
#                 attr = attr.to(self.device)
#         setattr(self, name, attr)

#     # @torch.no_grad()
#     def sample(self,
#                S, # sample timestep
#                batch_size,
#                shape,
#                conditioning=None,
#                callback=None,
#                img_callback=None,
#                quantize_x0=False,
#                order=1,
#                skip_type='quad',
#                method='singlestep',
#                guidance_type='classifier-free',
#                x0=None,
#                noise_dropout=0.,
#                verbose=True,
#                x_T=None,
#                log_every_t=100,
#                unconditional_guidance_scale=1.,
#                unconditional_conditioning=None,
#                # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
#                **kwargs
#                ):
#         if conditioning is not None:
#             if isinstance(conditioning, dict):
#                 cbs = conditioning[list(conditioning.keys())[0]].shape[0]
#                 if cbs != batch_size:
#                     print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
#             else:
#                 if conditioning.shape[0] != batch_size:
#                     print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

#         to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.model.device)
#         self.register_buffer('betas', to_torch(self.model.betas))

#         # sampling
#         C, H, W = shape
#         shape = (batch_size, C, H, W)

#         device = self.model.betas.device
#         b = shape[0]
#         if x_T is None:
#             img = torch.randn(shape, device=device)
#         else:
#             img = x_T

#         print(f"use dpm-solver with {S} steps")
#         noise_schedule = NoiseScheduleVP(schedule='discrete', betas=self.betas)
#         model_fn = model_wrapper(
#             self.model,
#             noise_schedule,
#             model_type="noise",
#             guidance_type = guidance_type,
#             condition=conditioning,
#             unconditional_condition=unconditional_conditioning,
#             guidance_scale=unconditional_guidance_scale
#         )
#         dpm_solver = DPM_Solver(model_fn, noise_schedule, algorithm_type=self.algorithm_type)
#         img, intermediates, x_interms, t_interms = dpm_solver.sample(
#             img,
#             steps=S,
#             order=order,
#             skip_type="time_quadratic" if skip_type == 'quad' else 'time_uniform',
#             method="singlestep",
#             return_intermediate = True,
#         )

#         return img, intermediates

# dpm-solver 中通过调用 model_fn 实现单步噪声估计，但是 fixed 的条件，现在想把 model_wrapper 的功能融合进 dpm-solver 的单步采样，关键是 将 model_wrapper 的参数全部融合进去

#         model_fn = model_wrapper(
#             self.model, # ✅
#             noise_schedule, # ✅
#             model_type="noise",
#             guidance_type = guidance_type, # 写入 p_sampling 单步采样函数的参数中，不作为 dpm-solver内置属性
#             condition=conditioning,
#             unconditional_condition=unconditional_conditioning,
#             guidance_scale=unconditional_guidance_scale
#         )

class DpmSolverSampler(object):
    def __init__(
            self, 
            model,  
            # model_fn, # 根据指定的 condition 计算 noise epsilon
            algorithm_type="dpmsolver",
            model_type="noise",
            correcting_x0_fn=None,
            correcting_xt_fn=None,
            thresholding_max_val=1.,
            dynamic_thresholding_ratio=0.995,
            device=torch.device('cuda:0'),
            **kwargs):
        super().__init__()
        self.model = model
        self.model_type = model_type
        self.device = device
        self.ddpm_num_timesteps = self.model.num_timesteps
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.device)
        self.register_buffer('betas', to_torch(self.model.betas))
        self.noise_schedule = NoiseScheduleVP(schedule='discrete', betas=self.betas)
        
        # -----------------------------
        # self.model = lambda x, t: model_fn(x, t.expand((x.shape[0])))
        assert algorithm_type in ["dpmsolver", "dpmsolver++"]
        self.algorithm_type = algorithm_type
        if correcting_x0_fn == "dynamic_thresholding":
            self.correcting_x0_fn = self.dynamic_thresholding_fn
        else:
            self.correcting_x0_fn = correcting_x0_fn
        self.correcting_xt_fn = correcting_xt_fn
        self.dynamic_thresholding_ratio = dynamic_thresholding_ratio
        self.thresholding_max_val = thresholding_max_val

    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != self.device:
                attr = attr.to(self.device)
        setattr(self, name, attr)

    def dynamic_thresholding_fn(self, x0, t):
        """
        The dynamic thresholding method. 
        """
        dims = x0.dim()
        p = self.dynamic_thresholding_ratio
        s = torch.quantile(torch.abs(x0).reshape((x0.shape[0], -1)), p, dim=1)
        s = expand_dims(torch.maximum(s, self.thresholding_max_val * torch.ones_like(s).to(s.device)), dims)
        x0 = torch.clamp(x0, -s, s) / s
        return x0

    def get_model_input_time(self, t_continuous):
        """
        Convert the continuous-time `t_continuous` (in [epsilon, T]) to the model input time.
        For discrete-time DPMs, we convert `t_continuous` in [1 / N, 1] to `t_input` in [0, 1000 * (N - 1) / N].
        For continuous-time DPMs, we just use `t_continuous`.
        """
        if self.noise_schedule.schedule == 'discrete':
            return (t_continuous - 1. / self.noise_schedule.total_N) * 1000.
        else:
            return t_continuous

    def noise_pred_fn(self, x, t_continuous, cond=None):
        t_input = self.get_model_input_time(t_continuous)
        output = self.model.apply_model(x, t_input, cond)
        if self.model_type == "noise":
            return output
        elif self.model_type == "x_start":
            alpha_t, sigma_t = self.noise_schedule.marginal_alpha(t_continuous), self.noise_schedule.marginal_std(t_continuous)
            return (x - alpha_t * output) / sigma_t
        elif self.model_type == "v":
            alpha_t, sigma_t = self.noise_schedule.marginal_alpha(t_continuous), self.noise_schedule.marginal_std(t_continuous)
            return alpha_t * output + sigma_t * x
        elif self.model_type == "score":
            sigma_t = self.noise_schedule.marginal_std(t_continuous)
            return -sigma_t * output

    def noise_prediction_fn(self, x, t, conditioning, unconditional_guidance_scale, unconditional_conditioning):
        """
        Return the noise prediction model.
        """
        t_continuous = t.expand((x.shape[0]))

        if unconditional_guidance_scale == 1. or unconditional_conditioning is None:
            return self.noise_pred_fn(x, t_continuous, cond=conditioning)
        else:
            x_in = torch.cat([x] * 2)
            t_in = torch.cat([t_continuous] * 2)
            c_in = torch.cat([unconditional_conditioning, conditioning])
            noise_uncond, noise = self.noise_pred_fn(x_in, t_in, cond=c_in).chunk(2)
            return noise_uncond + unconditional_guidance_scale * (noise - noise_uncond)

    def data_prediction_fn(self, x, t, conditioning, unconditional_guidance_scale, unconditional_conditioning):
        """
        Return the data prediction model (with corrector).
        """
        noise = self.noise_prediction_fn(x, t, conditioning, unconditional_guidance_scale, unconditional_conditioning)
        alpha_t, sigma_t = self.noise_schedule.marginal_alpha(t), self.noise_schedule.marginal_std(t)
        x0 = (x - sigma_t * noise) / alpha_t
        if self.correcting_x0_fn is not None:
            x0 = self.correcting_x0_fn(x0, t)
        return x0

    def model_fn(self, x, t, conditioning, unconditional_guidance_scale, unconditional_conditioning):
        """
        Convert the model to the noise prediction model or the data prediction model. 
        """
        if self.algorithm_type == "dpmsolver++":
            return self.data_prediction_fn(x, t, conditioning, unconditional_guidance_scale, unconditional_conditioning)
        else:
            return self.noise_prediction_fn(x, t, conditioning, unconditional_guidance_scale, unconditional_conditioning)

    def get_time_steps(self, skip_type, t_T, t_0, N, device):
        """Compute the intermediate time steps for sampling.

        Args:
            skip_type: A `str`. The type for the spacing of the time steps. We support three types:
                - 'logSNR': uniform logSNR for the time steps.
                - 'time_uniform': uniform time for the time steps. (**Recommended for high-resolutional data**.)
                - 'time_quadratic': quadratic time for the time steps. (Used in DDIM for low-resolutional data.)
            t_T: A `float`. The starting time of the sampling (default is T).
            t_0: A `float`. The ending time of the sampling (default is epsilon).
            N: A `int`. The total number of the spacing of the time steps.
            device: A torch device.
        Returns:
            A pytorch tensor of the time steps, with the shape (N + 1,).
        """
        if skip_type == 'logSNR':
            lambda_T = self.noise_schedule.marginal_lambda(torch.tensor(t_T).to(device))
            lambda_0 = self.noise_schedule.marginal_lambda(torch.tensor(t_0).to(device))
            logSNR_steps = torch.linspace(lambda_T.cpu().item(), lambda_0.cpu().item(), N + 1).to(device)
            return self.noise_schedule.inverse_lambda(logSNR_steps)
        elif skip_type == 'time_uniform':
            return torch.linspace(t_T, t_0, N + 1).to(device)
        elif skip_type == 'time_quadratic':
            t_order = 2
            t = torch.linspace(t_T**(1. / t_order), t_0**(1. / t_order), N + 1).pow(t_order).to(device)
            return t
        else:
            raise ValueError("Unsupported skip_type {}, need to be 'logSNR' or 'time_uniform' or 'time_quadratic'".format(skip_type))

    def get_orders_and_timesteps_for_singlestep_solver(self, steps, order, skip_type, t_T, t_0, device):
        """
        Get the order of each step for sampling by the singlestep DPM-Solver.

        We combine both DPM-Solver-1,2,3 to use all the function evaluations, which is named as "DPM-Solver-fast".
        Given a fixed number of function evaluations by `steps`, the sampling procedure by DPM-Solver-fast is:
            - If order == 1:
                We take `steps` of DPM-Solver-1 (i.e. DDIM).
            - If order == 2:
                - Denote K = (steps // 2). We take K or (K + 1) intermediate time steps for sampling.
                - If steps % 2 == 0, we use K steps of DPM-Solver-2.
                - If steps % 2 == 1, we use K steps of DPM-Solver-2 and 1 step of DPM-Solver-1.
            - If order == 3:
                - Denote K = (steps // 3 + 1). We take K intermediate time steps for sampling.
                - If steps % 3 == 0, we use (K - 2) steps of DPM-Solver-3, and 1 step of DPM-Solver-2 and 1 step of DPM-Solver-1.
                - If steps % 3 == 1, we use (K - 1) steps of DPM-Solver-3 and 1 step of DPM-Solver-1.
                - If steps % 3 == 2, we use (K - 1) steps of DPM-Solver-3 and 1 step of DPM-Solver-2.

        ============================================
        Args:
            order: A `int`. The max order for the solver (2 or 3).
            steps: A `int`. The total number of function evaluations (NFE).
            skip_type: A `str`. The type for the spacing of the time steps. We support three types:
                - 'logSNR': uniform logSNR for the time steps.
                - 'time_uniform': uniform time for the time steps. (**Recommended for high-resolutional data**.)
                - 'time_quadratic': quadratic time for the time steps. (Used in DDIM for low-resolutional data.)
            t_T: A `float`. The starting time of the sampling (default is T).
            t_0: A `float`. The ending time of the sampling (default is epsilon).
            device: A torch device.
        Returns:
            orders: A list of the solver order of each step.
        """
        if order == 3:
            K = steps // 3 + 1
            if steps % 3 == 0:
                orders = [3,] * (K - 2) + [2, 1]
            elif steps % 3 == 1:
                orders = [3,] * (K - 1) + [1]
            else:
                orders = [3,] * (K - 1) + [2]
        elif order == 2:
            if steps % 2 == 0:
                K = steps // 2
                orders = [2,] * K
            else:
                K = steps // 2 + 1
                orders = [2,] * (K - 1) + [1]
        elif order == 1:
            K = 1
            orders = [1,] * steps
        else:
            raise ValueError("'order' must be '1' or '2' or '3'.")
        if skip_type == 'logSNR':
            # To reproduce the results in DPM-Solver paper
            timesteps_outer = self.get_time_steps(skip_type, t_T, t_0, K, device)
        else:
            timesteps_outer = self.get_time_steps(skip_type, t_T, t_0, steps, device)[torch.cumsum(torch.tensor([0,] + orders), 0).to(device)]
        return timesteps_outer, orders

    def denoise_to_zero_fn(self, x, s):
        """
        Denoise at the final step, which is equivalent to solve the ODE from lambda_s to infty by first-order discretization. 
        """
        return self.data_prediction_fn(x, s)

    def dpm_solver_first_update(self, x, s, t, model_s=None, return_intermediate=False, conditioning=None, log_every_t=100, 
                                unconditional_guidance_scale=1., unconditional_conditioning=None):
        """
        DPM-Solver-1 (equivalent to DDIM) from time `s` to time `t`.

        Args:
            x: A pytorch tensor. The initial value at time `s`.
            s: A pytorch tensor. The starting time, with the shape (1,).
            t: A pytorch tensor. The ending time, with the shape (1,).
            model_s: A pytorch tensor. The model function evaluated at time `s`.
                If `model_s` is None, we evaluate the model by `x` and `s`; otherwise we directly use it.
            return_intermediate: A `bool`. If true, also return the model value at time `s`.
        Returns:
            x_t: A pytorch tensor. The approximated solution at time `t`.
        """
        ns = self.noise_schedule
        dims = x.dim()
        lambda_s, lambda_t = ns.marginal_lambda(s), ns.marginal_lambda(t)
        h = lambda_t - lambda_s
        log_alpha_s, log_alpha_t = ns.marginal_log_mean_coeff(s), ns.marginal_log_mean_coeff(t)
        sigma_s, sigma_t = ns.marginal_std(s), ns.marginal_std(t)
        alpha_t = torch.exp(log_alpha_t)

        if self.algorithm_type == "dpmsolver++":
            phi_1 = torch.expm1(-h)
            if model_s is None:
                model_s = self.model_fn(x, s,
                                    conditioning=conditioning, 
                                    unconditional_guidance_scale=unconditional_guidance_scale, 
                                    unconditional_conditioning=unconditional_conditioning)
            x_t = (
                sigma_t / sigma_s * x
                - alpha_t * phi_1 * model_s
            )
            if return_intermediate:
                return x_t, {'model_s': model_s}
            else:
                return x_t
        else:
            phi_1 = torch.expm1(h)
            if model_s is None:
                model_s = self.model_fn(x, s,
                                    conditioning=conditioning, 
                                    unconditional_guidance_scale=unconditional_guidance_scale, 
                                    unconditional_conditioning=unconditional_conditioning)
            x_t = (
                torch.exp(log_alpha_t - log_alpha_s) * x
                - (sigma_t * phi_1) * model_s
            )
            if return_intermediate:
                return x_t, {'model_s': model_s}
            else:
                return x_t

    def singlestep_dpm_solver_second_update(self, x, s, t, r1=0.5, model_s=None, return_intermediate=False, solver_type='dpmsolver',
                                            conditioning=None, log_every_t=100, unconditional_guidance_scale=1., unconditional_conditioning=None):
        """
        Singlestep solver DPM-Solver-2 from time `s` to time `t`.

        Args:
            x: A pytorch tensor. The initial value at time `s`.
            s: A pytorch tensor. The starting time, with the shape (1,).
            t: A pytorch tensor. The ending time, with the shape (1,).
            r1: A `float`. The hyperparameter of the second-order solver.
            model_s: A pytorch tensor. The model function evaluated at time `s`.
                If `model_s` is None, we evaluate the model by `x` and `s`; otherwise we directly use it.
            return_intermediate: A `bool`. If true, also return the model value at time `s` and `s1` (the intermediate time).
            solver_type: either 'dpmsolver' or 'taylor'. The type for the high-order solvers.
                The type slightly impacts the performance. We recommend to use 'dpmsolver' type.
        Returns:
            x_t: A pytorch tensor. The approximated solution at time `t`.
        """
        if solver_type not in ['dpmsolver', 'taylor']:
            raise ValueError("'solver_type' must be either 'dpmsolver' or 'taylor', got {}".format(solver_type))
        if r1 is None:
            r1 = 0.5
        ns = self.noise_schedule
        lambda_s, lambda_t = ns.marginal_lambda(s), ns.marginal_lambda(t)
        h = lambda_t - lambda_s
        lambda_s1 = lambda_s + r1 * h
        s1 = ns.inverse_lambda(lambda_s1)
        log_alpha_s, log_alpha_s1, log_alpha_t = ns.marginal_log_mean_coeff(s), ns.marginal_log_mean_coeff(s1), ns.marginal_log_mean_coeff(t)
        sigma_s, sigma_s1, sigma_t = ns.marginal_std(s), ns.marginal_std(s1), ns.marginal_std(t)
        alpha_s1, alpha_t = torch.exp(log_alpha_s1), torch.exp(log_alpha_t)

        if self.algorithm_type == "dpmsolver++":
            phi_11 = torch.expm1(-r1 * h)
            phi_1 = torch.expm1(-h)

            if model_s is None:
                model_s = self.model_fn(x, s, conditioning=conditioning, unconditional_guidance_scale=unconditional_guidance_scale, unconditional_conditioning=unconditional_conditioning)
            x_s1 = (
                (sigma_s1 / sigma_s) * x
                - (alpha_s1 * phi_11) * model_s
            )
            model_s1 = self.model_fn(x_s1, s1, conditioning=conditioning, unconditional_guidance_scale=unconditional_guidance_scale, unconditional_conditioning=unconditional_conditioning)
            if solver_type == 'dpmsolver':
                x_t = (
                    (sigma_t / sigma_s) * x
                    - (alpha_t * phi_1) * model_s
                    - (0.5 / r1) * (alpha_t * phi_1) * (model_s1 - model_s)
                )
            elif solver_type == 'taylor':
                x_t = (
                    (sigma_t / sigma_s) * x
                    - (alpha_t * phi_1) * model_s
                    + (1. / r1) * (alpha_t * (phi_1 / h + 1.)) * (model_s1 - model_s)
                )
        else:
            phi_11 = torch.expm1(r1 * h)
            phi_1 = torch.expm1(h)

            if model_s is None:
                model_s = self.model_fn(x, s, conditioning=conditioning, unconditional_guidance_scale=unconditional_guidance_scale, unconditional_conditioning=unconditional_conditioning)
            x_s1 = (
                torch.exp(log_alpha_s1 - log_alpha_s) * x
                - (sigma_s1 * phi_11) * model_s
            )
            model_s1 = self.model_fn(x_s1, s1, conditioning=conditioning, unconditional_guidance_scale=unconditional_guidance_scale, unconditional_conditioning=unconditional_conditioning)
            if solver_type == 'dpmsolver':
                x_t = (
                    torch.exp(log_alpha_t - log_alpha_s) * x
                    - (sigma_t * phi_1) * model_s
                    - (0.5 / r1) * (sigma_t * phi_1) * (model_s1 - model_s)
                )
            elif solver_type == 'taylor':
                x_t = (
                    torch.exp(log_alpha_t - log_alpha_s) * x
                    - (sigma_t * phi_1) * model_s
                    - (1. / r1) * (sigma_t * (phi_1 / h - 1.)) * (model_s1 - model_s)
                )

        
        if return_intermediate:
            return x_t, {'model_s': model_s, 'model_s1': model_s1, 'x_s1': x_s1, 's1': s1}
        else:
            return x_t

    def singlestep_dpm_solver_third_update(self, x, s, t, r1=1./3., r2=2./3., model_s=None, model_s1=None, return_intermediate=False, solver_type='dpmsolver',
                                           conditioning=None, log_every_t=100, unconditional_guidance_scale=1., unconditional_conditioning=None):
        """
        Singlestep solver DPM-Solver-3 from time `s` to time `t`.

        Args:
            x: A pytorch tensor. The initial value at time `s`.
            s: A pytorch tensor. The starting time, with the shape (1,).
            t: A pytorch tensor. The ending time, with the shape (1,).
            r1: A `float`. The hyperparameter of the third-order solver.
            r2: A `float`. The hyperparameter of the third-order solver.
            model_s: A pytorch tensor. The model function evaluated at time `s`.
                If `model_s` is None, we evaluate the model by `x` and `s`; otherwise we directly use it.
            model_s1: A pytorch tensor. The model function evaluated at time `s1` (the intermediate time given by `r1`).
                If `model_s1` is None, we evaluate the model at `s1`; otherwise we directly use it.
            return_intermediate: A `bool`. If true, also return the model value at time `s`, `s1` and `s2` (the intermediate times).
            solver_type: either 'dpmsolver' or 'taylor'. The type for the high-order solvers.
                The type slightly impacts the performance. We recommend to use 'dpmsolver' type.
        Returns:
            x_t: A pytorch tensor. The approximated solution at time `t`.
        """
        if solver_type not in ['dpmsolver', 'taylor']:
            raise ValueError("'solver_type' must be either 'dpmsolver' or 'taylor', got {}".format(solver_type))
        if r1 is None:
            r1 = 1. / 3.
        if r2 is None:
            r2 = 2. / 3.
        ns = self.noise_schedule
        lambda_s, lambda_t = ns.marginal_lambda(s), ns.marginal_lambda(t)
        h = lambda_t - lambda_s
        lambda_s1 = lambda_s + r1 * h
        lambda_s2 = lambda_s + r2 * h
        s1 = ns.inverse_lambda(lambda_s1)
        s2 = ns.inverse_lambda(lambda_s2)
        log_alpha_s, log_alpha_s1, log_alpha_s2, log_alpha_t = ns.marginal_log_mean_coeff(s), ns.marginal_log_mean_coeff(s1), ns.marginal_log_mean_coeff(s2), ns.marginal_log_mean_coeff(t)
        sigma_s, sigma_s1, sigma_s2, sigma_t = ns.marginal_std(s), ns.marginal_std(s1), ns.marginal_std(s2), ns.marginal_std(t)
        alpha_s1, alpha_s2, alpha_t = torch.exp(log_alpha_s1), torch.exp(log_alpha_s2), torch.exp(log_alpha_t)

        if self.algorithm_type == "dpmsolver++":
            phi_11 = torch.expm1(-r1 * h)
            phi_12 = torch.expm1(-r2 * h)
            phi_1 = torch.expm1(-h)
            phi_22 = torch.expm1(-r2 * h) / (r2 * h) + 1.
            phi_2 = phi_1 / h + 1.
            phi_3 = phi_2 / h - 0.5

            if model_s is None:
                model_s = self.model_fn(x, s, conditioning=conditioning, 
                                        unconditional_guidance_scale=unconditional_guidance_scale, 
                                        unconditional_conditioning=unconditional_conditioning)
            if model_s1 is None:
                x_s1 = (
                    (sigma_s1 / sigma_s) * x
                    - (alpha_s1 * phi_11) * model_s
                )
                model_s1 = self.model_fn(x_s1, s1, conditioning=conditioning, 
                                        unconditional_guidance_scale=unconditional_guidance_scale, 
                                        unconditional_conditioning=unconditional_conditioning)
            x_s2 = (
                (sigma_s2 / sigma_s) * x
                - (alpha_s2 * phi_12) * model_s
                + r2 / r1 * (alpha_s2 * phi_22) * (model_s1 - model_s)
            )
            model_s2 = self.model_fn(x_s2, s2,
                                    conditioning=conditioning, 
                                    unconditional_guidance_scale=unconditional_guidance_scale, 
                                    unconditional_conditioning=unconditional_conditioning)
            if solver_type == 'dpmsolver':
                x_t = (
                    (sigma_t / sigma_s) * x
                    - (alpha_t * phi_1) * model_s
                    + (1. / r2) * (alpha_t * phi_2) * (model_s2 - model_s)
                )
            elif solver_type == 'taylor':
                D1_0 = (1. / r1) * (model_s1 - model_s)
                D1_1 = (1. / r2) * (model_s2 - model_s)
                D1 = (r2 * D1_0 - r1 * D1_1) / (r2 - r1)
                D2 = 2. * (D1_1 - D1_0) / (r2 - r1)
                x_t = (
                    (sigma_t / sigma_s) * x
                    - (alpha_t * phi_1) * model_s
                    + (alpha_t * phi_2) * D1
                    - (alpha_t * phi_3) * D2
                )
        else:
            phi_11 = torch.expm1(r1 * h)
            phi_12 = torch.expm1(r2 * h)
            phi_1 = torch.expm1(h)
            phi_22 = torch.expm1(r2 * h) / (r2 * h) - 1.
            phi_2 = phi_1 / h - 1.
            phi_3 = phi_2 / h - 0.5

            if model_s is None:
                model_s = self.model_fn(x, s,
                                        conditioning=conditioning, 
                                        unconditional_guidance_scale=unconditional_guidance_scale, 
                                        unconditional_conditioning=unconditional_conditioning)
            if model_s1 is None:
                x_s1 = (
                    (torch.exp(log_alpha_s1 - log_alpha_s)) * x
                    - (sigma_s1 * phi_11) * model_s
                )
                model_s1 = self.model_fn(x_s1, s1,
                                        conditioning=conditioning, 
                                        unconditional_guidance_scale=unconditional_guidance_scale, 
                                        unconditional_conditioning=unconditional_conditioning)
            x_s2 = (
                (torch.exp(log_alpha_s2 - log_alpha_s)) * x
                - (sigma_s2 * phi_12) * model_s
                - r2 / r1 * (sigma_s2 * phi_22) * (model_s1 - model_s)
            )
            model_s2 = self.model_fn(x_s2, s2,
                                    conditioning=conditioning, 
                                    unconditional_guidance_scale=unconditional_guidance_scale, 
                                    unconditional_conditioning=unconditional_conditioning)
            if solver_type == 'dpmsolver':
                x_t = (
                    (torch.exp(log_alpha_t - log_alpha_s)) * x
                    - (sigma_t * phi_1) * model_s
                    - (1. / r2) * (sigma_t * phi_2) * (model_s2 - model_s)
                )
            elif solver_type == 'taylor':
                D1_0 = (1. / r1) * (model_s1 - model_s)
                D1_1 = (1. / r2) * (model_s2 - model_s)
                D1 = (r2 * D1_0 - r1 * D1_1) / (r2 - r1)
                D2 = 2. * (D1_1 - D1_0) / (r2 - r1)
                x_t = (
                    (torch.exp(log_alpha_t - log_alpha_s)) * x
                    - (sigma_t * phi_1) * model_s
                    - (sigma_t * phi_2) * D1
                    - (sigma_t * phi_3) * D2
                )

        if return_intermediate:
            return x_t, {'model_s': model_s, 'model_s1': model_s1, 'model_s2': model_s2}
        else:
            return x_t

    def multistep_dpm_solver_second_update(self, x, model_prev_list, t_prev_list, t, solver_type="dpmsolver"):
        """
        Multistep solver DPM-Solver-2 from time `t_prev_list[-1]` to time `t`.

        Args:
            x: A pytorch tensor. The initial value at time `s`.
            model_prev_list: A list of pytorch tensor. The previous computed model values.
            t_prev_list: A list of pytorch tensor. The previous times, each time has the shape (1,)
            t: A pytorch tensor. The ending time, with the shape (1,).
            solver_type: either 'dpmsolver' or 'taylor'. The type for the high-order solvers.
                The type slightly impacts the performance. We recommend to use 'dpmsolver' type.
        Returns:
            x_t: A pytorch tensor. The approximated solution at time `t`.
        """
        if solver_type not in ['dpmsolver', 'taylor']:
            raise ValueError("'solver_type' must be either 'dpmsolver' or 'taylor', got {}".format(solver_type))
        ns = self.noise_schedule
        model_prev_1, model_prev_0 = model_prev_list[-2], model_prev_list[-1]
        t_prev_1, t_prev_0 = t_prev_list[-2], t_prev_list[-1]
        lambda_prev_1, lambda_prev_0, lambda_t = ns.marginal_lambda(t_prev_1), ns.marginal_lambda(t_prev_0), ns.marginal_lambda(t)
        log_alpha_prev_0, log_alpha_t = ns.marginal_log_mean_coeff(t_prev_0), ns.marginal_log_mean_coeff(t)
        sigma_prev_0, sigma_t = ns.marginal_std(t_prev_0), ns.marginal_std(t)
        alpha_t = torch.exp(log_alpha_t)

        h_0 = lambda_prev_0 - lambda_prev_1
        h = lambda_t - lambda_prev_0
        r0 = h_0 / h
        D1_0 = (1. / r0) * (model_prev_0 - model_prev_1)
        if self.algorithm_type == "dpmsolver++":
            phi_1 = torch.expm1(-h)
            if solver_type == 'dpmsolver':
                x_t = (
                    (sigma_t / sigma_prev_0) * x
                    - (alpha_t * phi_1) * model_prev_0
                    - 0.5 * (alpha_t * phi_1) * D1_0
                )
            elif solver_type == 'taylor':
                x_t = (
                    (sigma_t / sigma_prev_0) * x
                    - (alpha_t * phi_1) * model_prev_0
                    + (alpha_t * (phi_1 / h + 1.)) * D1_0
                )
        else:
            phi_1 = torch.expm1(h)
            if solver_type == 'dpmsolver':
                x_t = (
                    (torch.exp(log_alpha_t - log_alpha_prev_0)) * x
                    - (sigma_t * phi_1) * model_prev_0
                    - 0.5 * (sigma_t * phi_1) * D1_0
                )
            elif solver_type == 'taylor':
                x_t = (
                    (torch.exp(log_alpha_t - log_alpha_prev_0)) * x
                    - (sigma_t * phi_1) * model_prev_0
                    - (sigma_t * (phi_1 / h - 1.)) * D1_0
                )
        return x_t

    def multistep_dpm_solver_third_update(self, x, model_prev_list, t_prev_list, t, solver_type='dpmsolver'):
        """
        Multistep solver DPM-Solver-3 from time `t_prev_list[-1]` to time `t`.

        Args:
            x: A pytorch tensor. The initial value at time `s`.
            model_prev_list: A list of pytorch tensor. The previous computed model values.
            t_prev_list: A list of pytorch tensor. The previous times, each time has the shape (1,)
            t: A pytorch tensor. The ending time, with the shape (1,).
            solver_type: either 'dpmsolver' or 'taylor'. The type for the high-order solvers.
                The type slightly impacts the performance. We recommend to use 'dpmsolver' type.
        Returns:
            x_t: A pytorch tensor. The approximated solution at time `t`.
        """
        ns = self.noise_schedule
        model_prev_2, model_prev_1, model_prev_0 = model_prev_list
        t_prev_2, t_prev_1, t_prev_0 = t_prev_list
        lambda_prev_2, lambda_prev_1, lambda_prev_0, lambda_t = ns.marginal_lambda(t_prev_2), ns.marginal_lambda(t_prev_1), ns.marginal_lambda(t_prev_0), ns.marginal_lambda(t)
        log_alpha_prev_0, log_alpha_t = ns.marginal_log_mean_coeff(t_prev_0), ns.marginal_log_mean_coeff(t)
        sigma_prev_0, sigma_t = ns.marginal_std(t_prev_0), ns.marginal_std(t)
        alpha_t = torch.exp(log_alpha_t)

        h_1 = lambda_prev_1 - lambda_prev_2
        h_0 = lambda_prev_0 - lambda_prev_1
        h = lambda_t - lambda_prev_0
        r0, r1 = h_0 / h, h_1 / h
        D1_0 = (1. / r0) * (model_prev_0 - model_prev_1)
        D1_1 = (1. / r1) * (model_prev_1 - model_prev_2)
        D1 = D1_0 + (r0 / (r0 + r1)) * (D1_0 - D1_1)
        D2 = (1. / (r0 + r1)) * (D1_0 - D1_1)
        if self.algorithm_type == "dpmsolver++":
            phi_1 = torch.expm1(-h)
            phi_2 = phi_1 / h + 1.
            phi_3 = phi_2 / h - 0.5
            x_t = (
                (sigma_t / sigma_prev_0) * x
                - (alpha_t * phi_1) * model_prev_0
                + (alpha_t * phi_2) * D1
                - (alpha_t * phi_3) * D2
            )
        else:
            phi_1 = torch.expm1(h)
            phi_2 = phi_1 / h - 1.
            phi_3 = phi_2 / h - 0.5
            x_t = (
                (torch.exp(log_alpha_t - log_alpha_prev_0)) * x
                - (sigma_t * phi_1) * model_prev_0
                - (sigma_t * phi_2) * D1
                - (sigma_t * phi_3) * D2
            )
        return x_t

    def singlestep_dpm_solver_update(self, x, s, t, order, solver_type='dpmsolver', r1=None, r2=None,  
                                    conditioning=None, log_every_t=100, unconditional_guidance_scale=1., 
                                    unconditional_conditioning=None, return_intermediate=False):
        """
        Singlestep DPM-Solver with the order `order` from time `s` to time `t`.

        Args:
            x: A pytorch tensor. The initial value at time `s`.
            s: A pytorch tensor. The starting time, with the shape (1,).
            t: A pytorch tensor. The ending time, with the shape (1,).
            order: A `int`. The order of DPM-Solver. We only support order == 1 or 2 or 3.
            return_intermediate: A `bool`. If true, also return the model value at time `s`, `s1` and `s2` (the intermediate times).
            solver_type: either 'dpmsolver' or 'taylor'. The type for the high-order solvers.
                The type slightly impacts the performance. We recommend to use 'dpmsolver' type.
            r1: A `float`. The hyperparameter of the second-order or third-order solver.
            r2: A `float`. The hyperparameter of the third-order solver.
        Returns:
            x_t: A pytorch tensor. The approximated solution at time `t`.
        """
        if order == 1:
            return self.dpm_solver_first_update(x, s, t, conditioning=conditioning, log_every_t=log_every_t, 
                                                unconditional_guidance_scale=unconditional_guidance_scale, 
                                                unconditional_conditioning=unconditional_conditioning,
                                                return_intermediate=return_intermediate)
        elif order == 2:
            return self.singlestep_dpm_solver_second_update(x, s, t, return_intermediate=return_intermediate, 
                                                            solver_type=solver_type, r1=r1,
                                                            conditioning=conditioning, log_every_t=log_every_t, 
                                                            unconditional_guidance_scale=unconditional_guidance_scale, 
                                                            unconditional_conditioning=unconditional_conditioning)
        elif order == 3:
            return self.singlestep_dpm_solver_third_update(x, s, t, return_intermediate=return_intermediate, 
                                                            solver_type=solver_type, r1=r1, r2=r2,
                                                            conditioning=conditioning, log_every_t=log_every_t, 
                                                            unconditional_guidance_scale=unconditional_guidance_scale, 
                                                            unconditional_conditioning=unconditional_conditioning)
        else:
            raise ValueError("Solver order must be 1 or 2 or 3, got {}".format(order))

    def multistep_dpm_solver_update(self, x, model_prev_list, t_prev_list, t, order, solver_type='dpmsolver'):
        """
        Multistep DPM-Solver with the order `order` from time `t_prev_list[-1]` to time `t`.

        Args:
            x: A pytorch tensor. The initial value at time `s`.
            model_prev_list: A list of pytorch tensor. The previous computed model values.
            t_prev_list: A list of pytorch tensor. The previous times, each time has the shape (1,)
            t: A pytorch tensor. The ending time, with the shape (1,).
            order: A `int`. The order of DPM-Solver. We only support order == 1 or 2 or 3.
            solver_type: either 'dpmsolver' or 'taylor'. The type for the high-order solvers.
                The type slightly impacts the performance. We recommend to use 'dpmsolver' type.
        Returns:
            x_t: A pytorch tensor. The approximated solution at time `t`.
        """
        if order == 1:
            return self.dpm_solver_first_update(x, t_prev_list[-1], t, model_s=model_prev_list[-1])
        elif order == 2:
            return self.multistep_dpm_solver_second_update(x, model_prev_list, t_prev_list, t, solver_type=solver_type)
        elif order == 3:
            return self.multistep_dpm_solver_third_update(x, model_prev_list, t_prev_list, t, solver_type=solver_type)
        else:
            raise ValueError("Solver order must be 1 or 2 or 3, got {}".format(order))

    def dpm_solver_adaptive(self, x, order, t_T, t_0, h_init=0.05, atol=0.0078, rtol=0.05, theta=0.9, t_err=1e-5, solver_type='dpmsolver'):
        """
        The adaptive step size solver based on singlestep DPM-Solver.

        Args:
            x: A pytorch tensor. The initial value at time `t_T`.
            order: A `int`. The (higher) order of the solver. We only support order == 2 or 3.
            t_T: A `float`. The starting time of the sampling (default is T).
            t_0: A `float`. The ending time of the sampling (default is epsilon).
            h_init: A `float`. The initial step size (for logSNR).
            atol: A `float`. The absolute tolerance of the solver. For image data, the default setting is 0.0078, followed [1].
            rtol: A `float`. The relative tolerance of the solver. The default setting is 0.05.
            theta: A `float`. The safety hyperparameter for adapting the step size. The default setting is 0.9, followed [1].
            t_err: A `float`. The tolerance for the time. We solve the diffusion ODE until the absolute error between the 
                current time and `t_0` is less than `t_err`. The default setting is 1e-5.
            solver_type: either 'dpmsolver' or 'taylor'. The type for the high-order solvers.
                The type slightly impacts the performance. We recommend to use 'dpmsolver' type.
        Returns:
            x_0: A pytorch tensor. The approximated solution at time `t_0`.

        [1] A. Jolicoeur-Martineau, K. Li, R. Piché-Taillefer, T. Kachman, and I. Mitliagkas, "Gotta go fast when generating data with score-based models," arXiv preprint arXiv:2105.14080, 2021.
        """
        ns = self.noise_schedule
        s = t_T * torch.ones((1,)).to(x)
        lambda_s = ns.marginal_lambda(s)
        lambda_0 = ns.marginal_lambda(t_0 * torch.ones_like(s).to(x))
        h = h_init * torch.ones_like(s).to(x)
        x_prev = x
        nfe = 0
        if order == 2:
            r1 = 0.5
            lower_update = lambda x, s, t: self.dpm_solver_first_update(x, s, t, return_intermediate=True)
            higher_update = lambda x, s, t, **kwargs: self.singlestep_dpm_solver_second_update(x, s, t, r1=r1, solver_type=solver_type, **kwargs)
        elif order == 3:
            r1, r2 = 1. / 3., 2. / 3.
            lower_update = lambda x, s, t: self.singlestep_dpm_solver_second_update(x, s, t, r1=r1, return_intermediate=True, solver_type=solver_type)
            higher_update = lambda x, s, t, **kwargs: self.singlestep_dpm_solver_third_update(x, s, t, r1=r1, r2=r2, solver_type=solver_type, **kwargs)
        else:
            raise ValueError("For adaptive step size solver, order must be 2 or 3, got {}".format(order))
        while torch.abs((s - t_0)).mean() > t_err:
            t = ns.inverse_lambda(lambda_s + h)
            x_lower, lower_noise_kwargs = lower_update(x, s, t)
            x_higher = higher_update(x, s, t, **lower_noise_kwargs)
            delta = torch.max(torch.ones_like(x).to(x) * atol, rtol * torch.max(torch.abs(x_lower), torch.abs(x_prev)))
            norm_fn = lambda v: torch.sqrt(torch.square(v.reshape((v.shape[0], -1))).mean(dim=-1, keepdim=True))
            E = norm_fn((x_higher - x_lower) / delta).max()
            if torch.all(E <= 1.):
                x = x_higher
                s = t
                x_prev = x_lower
                lambda_s = ns.marginal_lambda(s)
            h = torch.min(theta * h * torch.float_power(E, -1. / order).float(), lambda_0 - lambda_s)
            nfe += order
        print('adaptive solver nfe', nfe)
        return x

    def add_noise(self, x, t, noise=None):
        """
        Compute the noised input xt = alpha_t * x + sigma_t * noise. 

        Args:
            x: A `torch.Tensor` with shape `(batch_size, *shape)`.
            t: A `torch.Tensor` with shape `(t_size,)`.
        Returns:
            xt with shape `(t_size, batch_size, *shape)`.
        """
        alpha_t, sigma_t = self.noise_schedule.marginal_alpha(t), self.noise_schedule.marginal_std(t)
        if noise is None:
            noise = torch.randn((t.shape[0], *x.shape), device=x.device)
        x = x.reshape((-1, *x.shape))
        xt = expand_dims(alpha_t, x.dim()) * x + expand_dims(sigma_t, x.dim()) * noise
        if t.shape[0] == 1:
            return xt.squeeze(0)
        else:
            return xt

    def inverse(self, x, steps=20, t_start=None, t_end=None, order=2, skip_type='time_uniform',
        method='multistep', lower_order_final=True, denoise_to_zero=False, solver_type='dpmsolver',
        atol=0.0078, rtol=0.05, return_intermediate=False,
    ):
        """
        Inverse the sample `x` from time `t_start` to `t_end` by DPM-Solver.
        For discrete-time DPMs, we use `t_start=1/N`, where `N` is the total time steps during training.
        """
        t_0 = 1. / self.noise_schedule.total_N if t_start is None else t_start
        t_T = self.noise_schedule.T if t_end is None else t_end
        assert t_0 > 0 and t_T > 0, "Time range needs to be greater than 0. For discrete-time DPMs, it needs to be in [1 / N, 1], where N is the length of betas array"
        return self.sample(x, steps=steps, t_start=t_0, t_end=t_T, order=order, skip_type=skip_type,
            method=method, lower_order_final=lower_order_final, denoise_to_zero=denoise_to_zero, solver_type=solver_type,
            atol=atol, rtol=rtol, return_intermediate=return_intermediate)

    def sample_loop(self, x, conditioning, steps=20, t_start=None, t_end=None, order=2, skip_type='time_uniform',
        method='multistep', lower_order_final=True, denoise_to_zero=False, solver_type='dpmsolver',
        atol=0.0078, rtol=0.05, log_every_t=100, unconditional_guidance_scale=1., 
        unconditional_conditioning=None, return_intermediate=False, 
    ):
        """
        Compute the sample at time `t_end` by DPM-Solver, given the initial `x` at time `t_start`.

        =====================================================

        We support the following algorithms for both noise prediction model and data prediction model:
            - 'singlestep':
                Singlestep DPM-Solver (i.e. "DPM-Solver-fast" in the paper), which combines different orders of singlestep DPM-Solver. 
                We combine all the singlestep solvers with order <= `order` to use up all the function evaluations (steps).
                The total number of function evaluations (NFE) == `steps`.
                Given a fixed NFE == `steps`, the sampling procedure is:
                    - If `order` == 1:
                        - Denote K = steps. We use K steps of DPM-Solver-1 (i.e. DDIM).
                    - If `order` == 2:
                        - Denote K = (steps // 2) + (steps % 2). We take K intermediate time steps for sampling.
                        - If steps % 2 == 0, we use K steps of singlestep DPM-Solver-2.
                        - If steps % 2 == 1, we use (K - 1) steps of singlestep DPM-Solver-2 and 1 step of DPM-Solver-1.
                    - If `order` == 3:
                        - Denote K = (steps // 3 + 1). We take K intermediate time steps for sampling.
                        - If steps % 3 == 0, we use (K - 2) steps of singlestep DPM-Solver-3, and 1 step of singlestep DPM-Solver-2 and 1 step of DPM-Solver-1.
                        - If steps % 3 == 1, we use (K - 1) steps of singlestep DPM-Solver-3 and 1 step of DPM-Solver-1.
                        - If steps % 3 == 2, we use (K - 1) steps of singlestep DPM-Solver-3 and 1 step of singlestep DPM-Solver-2.
            - 'multistep':
                Multistep DPM-Solver with the order of `order`. The total number of function evaluations (NFE) == `steps`.
                We initialize the first `order` values by lower order multistep solvers.
                Given a fixed NFE == `steps`, the sampling procedure is:
                    Denote K = steps.
                    - If `order` == 1:
                        - We use K steps of DPM-Solver-1 (i.e. DDIM).
                    - If `order` == 2:
                        - We firstly use 1 step of DPM-Solver-1, then use (K - 1) step of multistep DPM-Solver-2.
                    - If `order` == 3:
                        - We firstly use 1 step of DPM-Solver-1, then 1 step of multistep DPM-Solver-2, then (K - 2) step of multistep DPM-Solver-3.
            - 'singlestep_fixed':
                Fixed order singlestep DPM-Solver (i.e. DPM-Solver-1 or singlestep DPM-Solver-2 or singlestep DPM-Solver-3).
                We use singlestep DPM-Solver-`order` for `order`=1 or 2 or 3, with total [`steps` // `order`] * `order` NFE.
            - 'adaptive':
                Adaptive step size DPM-Solver (i.e. "DPM-Solver-12" and "DPM-Solver-23" in the paper).
                We ignore `steps` and use adaptive step size DPM-Solver with a higher order of `order`.
                You can adjust the absolute tolerance `atol` and the relative tolerance `rtol` to balance the computatation costs
                (NFE) and the sample quality.
                    - If `order` == 2, we use DPM-Solver-12 which combines DPM-Solver-1 and singlestep DPM-Solver-2.
                    - If `order` == 3, we use DPM-Solver-23 which combines singlestep DPM-Solver-2 and singlestep DPM-Solver-3.

        =====================================================

        Some advices for choosing the algorithm:
            - For **unconditional sampling** or **guided sampling with small guidance scale** by DPMs:
                Use singlestep DPM-Solver or DPM-Solver++ ("DPM-Solver-fast" in the paper) with `order = 3`.
                e.g., DPM-Solver:
                    >>> dpm_solver = DPM_Solver(model_fn, noise_schedule, algorithm_type="dpmsolver")
                    >>> x_sample = dpm_solver.sample(x, steps=steps, t_start=t_start, t_end=t_end, order=3,
                            skip_type='time_uniform', method='singlestep')
                e.g., DPM-Solver++:
                    >>> dpm_solver = DPM_Solver(model_fn, noise_schedule, algorithm_type="dpmsolver++")
                    >>> x_sample = dpm_solver.sample(x, steps=steps, t_start=t_start, t_end=t_end, order=3,
                            skip_type='time_uniform', method='singlestep')
            - For **guided sampling with large guidance scale** by DPMs:
                Use multistep DPM-Solver with `algorithm_type="dpmsolver++"` and `order = 2`.
                e.g.
                    >>> dpm_solver = DPM_Solver(model_fn, noise_schedule, algorithm_type="dpmsolver++")
                    >>> x_sample = dpm_solver.sample(x, steps=steps, t_start=t_start, t_end=t_end, order=2,
                            skip_type='time_uniform', method='multistep')

        We support three types of `skip_type`:
            - 'logSNR': uniform logSNR for the time steps. **Recommended for low-resolutional images**
            - 'time_uniform': uniform time for the time steps. **Recommended for high-resolutional images**.
            - 'time_quadratic': quadratic time for the time steps.

        =====================================================
        Args:
            x: A pytorch tensor. The initial value at time `t_start`
                e.g. if `t_start` == T, then `x` is a sample from the standard normal distribution.
            steps: A `int`. The total number of function evaluations (NFE).
            t_start: A `float`. The starting time of the sampling.
                If `T` is None, we use self.noise_schedule.T (default is 1.0).
            t_end: A `float`. The ending time of the sampling.
                If `t_end` is None, we use 1. / self.noise_schedule.total_N.
                e.g. if total_N == 1000, we have `t_end` == 1e-3.
                For discrete-time DPMs:
                    - We recommend `t_end` == 1. / self.noise_schedule.total_N.
                For continuous-time DPMs:
                    - We recommend `t_end` == 1e-3 when `steps` <= 15; and `t_end` == 1e-4 when `steps` > 15.
            order: A `int`. The order of DPM-Solver.
            skip_type: A `str`. The type for the spacing of the time steps. 'time_uniform' or 'logSNR' or 'time_quadratic'.
            method: A `str`. The method for sampling. 'singlestep' or 'multistep' or 'singlestep_fixed' or 'adaptive'.
            denoise_to_zero: A `bool`. Whether to denoise to time 0 at the final step.
                Default is `False`. If `denoise_to_zero` is `True`, the total NFE is (`steps` + 1).

                This trick is firstly proposed by DDPM (https://arxiv.org/abs/2006.11239) and
                score_sde (https://arxiv.org/abs/2011.13456). Such trick can improve the FID
                for diffusion models sampling by diffusion SDEs for low-resolutional images
                (such as CIFAR-10). However, we observed that such trick does not matter for
                high-resolutional images. As it needs an additional NFE, we do not recommend
                it for high-resolutional images.
            lower_order_final: A `bool`. Whether to use lower order solvers at the final steps.
                Only valid for `method=multistep` and `steps < 15`. We empirically find that
                this trick is a key to stabilizing the sampling by DPM-Solver with very few steps
                (especially for steps <= 10). So we recommend to set it to be `True`.
            solver_type: A `str`. The taylor expansion type for the solver. `dpmsolver` or `taylor`. We recommend `dpmsolver`.
            atol: A `float`. The absolute tolerance of the adaptive step size solver. Valid when `method` == 'adaptive'.
            rtol: A `float`. The relative tolerance of the adaptive step size solver. Valid when `method` == 'adaptive'.
            return_intermediate: A `bool`. Whether to save the xt at each step.
                When set to `True`, method returns a tuple (x0, intermediates); when set to False, method returns only x0.
        Returns:
            x_end: A pytorch tensor. The approximated solution at time `t_end`.

        """
        t_0 = 1. / self.noise_schedule.total_N if t_end is None else t_end
        t_T = self.noise_schedule.T if t_start is None else t_start
        assert t_0 > 0 and t_T > 0, "Time range needs to be greater than 0. For discrete-time DPMs, it needs to be in [1 / N, 1], where N is the length of betas array"
        if return_intermediate:
            assert method in ['multistep', 'singlestep', 'singlestep_fixed'], "Cannot use adaptive solver when saving intermediate values"
        if self.correcting_xt_fn is not None:
            assert method in ['multistep', 'singlestep', 'singlestep_fixed'], "Cannot use adaptive solver when correcting_xt_fn is not None"
        device = x.device
        intermediates = []
        x_interms = []
        t_interms = []
        if order == 1:
            x_interms.append(x)

        with torch.no_grad():
            if method == 'adaptive':
                x = self.dpm_solver_adaptive(x, order=order, t_T=t_T, t_0=t_0, atol=atol, rtol=rtol, solver_type=solver_type)
            elif method == 'multistep':
                assert steps >= order
                timesteps = self.get_time_steps(skip_type=skip_type, t_T=t_T, t_0=t_0, N=steps, device=device)
                assert timesteps.shape[0] - 1 == steps
                # Init the initial values.
                step = 0
                t = timesteps[step]
                t_prev_list = [t]
                model_prev_list = [self.model_fn(x, t)]
                if self.correcting_xt_fn is not None:
                    x = self.correcting_xt_fn(x, t, step)
                if return_intermediate:
                    intermediates.append(x)
                # Init the first `order` values by lower order multistep DPM-Solver.
                for step in range(1, order):
                    t = timesteps[step]
                    x = self.multistep_dpm_solver_update(x, model_prev_list, t_prev_list, t, step, solver_type=solver_type)
                    if self.correcting_xt_fn is not None:
                        x = self.correcting_xt_fn(x, t, step)
                    if return_intermediate:
                        intermediates.append(x)
                    t_prev_list.append(t)
                    model_prev_list.append(self.model_fn(x, t))
                # Compute the remaining values by `order`-th order multistep DPM-Solver.
                for step in range(order, steps + 1):
                    t = timesteps[step]
                    # We only use lower order for steps < 10
                    if lower_order_final and steps < 10:
                        step_order = min(order, steps + 1 - step)
                    else:
                        step_order = order
                    x = self.multistep_dpm_solver_update(x, model_prev_list, t_prev_list, t, step_order, solver_type=solver_type)
                    if self.correcting_xt_fn is not None:
                        x = self.correcting_xt_fn(x, t, step)
                    if return_intermediate:
                        intermediates.append(x)
                    for i in range(order - 1):
                        t_prev_list[i] = t_prev_list[i + 1]
                        model_prev_list[i] = model_prev_list[i + 1]
                    t_prev_list[-1] = t
                    # We do not need to evaluate the final model value.
                    if step < steps:
                        model_prev_list[-1] = self.model_fn(x, t)
            elif method in ['singlestep', 'singlestep_fixed']:
                if method == 'singlestep':
                    timesteps_outer, orders = self.get_orders_and_timesteps_for_singlestep_solver(steps=steps, order=order, skip_type=skip_type, t_T=t_T, t_0=t_0, device=device)
                elif method == 'singlestep_fixed':
                    K = steps // order
                    orders = [order,] * K
                    timesteps_outer = self.get_time_steps(skip_type=skip_type, t_T=t_T, t_0=t_0, N=K, device=device)
                for step, order in enumerate(orders):
                    s, t = timesteps_outer[step], timesteps_outer[step + 1]
                    timesteps_inner = self.get_time_steps(skip_type=skip_type, t_T=s.item(), t_0=t.item(), N=order, device=device)
                    lambda_inner = self.noise_schedule.marginal_lambda(timesteps_inner)
                    h = lambda_inner[-1] - lambda_inner[0]
                    r1 = None if order <= 1 else (lambda_inner[1] - lambda_inner[0]) / h
                    r2 = None if order <= 2 else (lambda_inner[2] - lambda_inner[0]) / h
                    x, inter_dict = self.singlestep_dpm_solver_update(x, s, t, order, solver_type=solver_type, r1=r1, r2=r2, 
                                                                    conditioning=conditioning, log_every_t=log_every_t,
                                                                    unconditional_guidance_scale=unconditional_guidance_scale, 
                                                                    unconditional_conditioning=unconditional_conditioning,
                                                                    return_intermediate=return_intermediate)
                    if self.correcting_xt_fn is not None:
                        x = self.correcting_xt_fn(x, t, step)
                    if return_intermediate:
                        intermediates.append(x)
                        if order == 1:
                            x_interms.append(x)
                            t_interms.append(s)
                        elif order == 2:
                            x_interms.append(inter_dict['x_s1'])
                            t_interms.append(inter_dict['s1'])
            else:
                raise ValueError("Got wrong method {}".format(method))
            if denoise_to_zero:
                t = torch.ones((1,)).to(device) * t_0
                x = self.denoise_to_zero_fn(x, t)
                if self.correcting_xt_fn is not None:
                    x = self.correcting_xt_fn(x, t, step + 1)
                if return_intermediate:
                    intermediates.append(x)
        if return_intermediate:
            return x, intermediates, x_interms, t_interms
        else:
            return x, timesteps_outer


    @torch.no_grad()
    def sample(self,
               S,
               batch_size,
               shape,
               order=1,
               skip_type='quad',
               conditioning=None,
               method="singlestep",
               verbose=True,
               x_T=None,
               log_every_t=100,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None,
               return_intermediate=True,
               # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
               **kwargs
               ):
        if conditioning is not None:
            if isinstance(conditioning, dict):
                cbs = conditioning[list(conditioning.keys())[0]].shape[0]
                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
            else:
                if conditioning.shape[0] != batch_size:
                    print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

        # sampling
        C, H, W = shape
        shape = (batch_size, C, H, W)

        device = self.model.betas.device

        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T

        print(f"use dpm-solver with {S} steps")

        img, intermediates, x_interms, t_interms = self.sample_loop(
            img,
            conditioning,
            steps=S,
            order=order,
            skip_type="time_quadratic" if skip_type == 'quad' else 'time_uniform',
            method=method,
            log_every_t=log_every_t,
            unconditional_guidance_scale=unconditional_guidance_scale,
            unconditional_conditioning=unconditional_conditioning,
            return_intermediate=return_intermediate
        )

        return img, intermediates

    @torch.no_grad()
    def collect_intermediate_data(self,
               S,
               batch_size,
               shape,
               order=1,
               skip_type='quad',
               conditioning=None,
               method="singlestep",
               verbose=True,
               x_T=None,
               log_every_t=100,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None,
               return_intermediate=True,
               # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
               **kwargs
               ):
        if conditioning is not None:
            if isinstance(conditioning, dict):
                cbs = conditioning[list(conditioning.keys())[0]].shape[0]
                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
            else:
                if conditioning.shape[0] != batch_size:
                    print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

        # sampling
        C, H, W = shape
        shape = (batch_size, C, H, W)

        device = self.model.betas.device

        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T

        print(f"use dpm-solver with {S} steps")

        img, intermediates, x_interms, t_interms = self.sample_loop(
            img,
            conditioning,
            steps=S,
            order=order,
            skip_type="time_quadratic" if skip_type == 'quad' else 'time_uniform',
            method=method,
            log_every_t=log_every_t,
            unconditional_guidance_scale=unconditional_guidance_scale,
            unconditional_conditioning=unconditional_conditioning,
            return_intermediate=return_intermediate
        )

        xt_list, t_list = [], []

        if order == 1:
            xt_list.extend(x_interms[:-1])
        elif order == 2:
            xt_list.extend(x_interms)
        t_list.extend([torch.tensor(t.item(), device=self.device) for t in t_interms])

        return xt_list, t_list
    
class DpmSolverSampler_trainer(object):
    def __init__(
            self, 
            fp_model, 
            quant_model, 
            fp_order,
            quant_order,
            # model_fn, # 根据指定的 condition 计算 noise epsilon
            algorithm_type="dpmsolver",
            model_type="noise",
            lr_scheduler=None, 
            optimizer=None,
            correcting_x0_fn=None,
            correcting_xt_fn=None,
            thresholding_max_val=1.,
            dynamic_thresholding_ratio=0.995,
            device=torch.device('cuda:0'),
            **kwargs):
        super().__init__()
        self.fp_model = fp_model
        self.quant_model = quant_model
        self.fp_order = fp_order
        self.quant_order = quant_order

        self.model_type = model_type
        self.lr_scheduler = lr_scheduler
        self.optimizer = optimizer
        self.device = device
        self.ddpm_num_timesteps = self.fp_model.num_timesteps
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.device)
        self.register_buffer('betas', to_torch(self.fp_model.betas))
        self.noise_schedule = NoiseScheduleVP(schedule='discrete', betas=self.betas)
        
        # -----------------------------
        # self.model = lambda x, t: model_fn(x, t.expand((x.shape[0])))
        assert algorithm_type in ["dpmsolver", "dpmsolver++"]
        self.algorithm_type = algorithm_type
        if correcting_x0_fn == "dynamic_thresholding":
            self.correcting_x0_fn = self.dynamic_thresholding_fn
        else:
            self.correcting_x0_fn = correcting_x0_fn
        self.correcting_xt_fn = correcting_xt_fn
        self.dynamic_thresholding_ratio = dynamic_thresholding_ratio
        self.thresholding_max_val = thresholding_max_val

    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != self.device:
                attr = attr.to(self.device)
        setattr(self, name, attr)

    def dynamic_thresholding_fn(self, x0, t):
        """
        The dynamic thresholding method. 
        """
        dims = x0.dim()
        p = self.dynamic_thresholding_ratio
        s = torch.quantile(torch.abs(x0).reshape((x0.shape[0], -1)), p, dim=1)
        s = expand_dims(torch.maximum(s, self.thresholding_max_val * torch.ones_like(s).to(s.device)), dims)
        x0 = torch.clamp(x0, -s, s) / s
        return x0

    def get_model_input_time(self, t_continuous):
        """
        Convert the continuous-time `t_continuous` (in [epsilon, T]) to the model input time.
        For discrete-time DPMs, we convert `t_continuous` in [1 / N, 1] to `t_input` in [0, 1000 * (N - 1) / N].
        For continuous-time DPMs, we just use `t_continuous`.
        """
        if self.noise_schedule.schedule == 'discrete':
            return (t_continuous - 1. / self.noise_schedule.total_N) * 1000.
        else:
            return t_continuous

    def noise_pred_fn(self, x, t_continuous, quantized, cond=None):
        t_input = self.get_model_input_time(t_continuous)
        if quantized:
            output = self.quant_model.apply_model(x, t_input, cond)
        else:
            output = self.fp_model.apply_model(x, t_input, cond)
        if self.model_type == "noise":
            return output
        elif self.model_type == "x_start":
            alpha_t, sigma_t = self.noise_schedule.marginal_alpha(t_continuous), self.noise_schedule.marginal_std(t_continuous)
            return (x - alpha_t * output) / sigma_t
        elif self.model_type == "v":
            alpha_t, sigma_t = self.noise_schedule.marginal_alpha(t_continuous), self.noise_schedule.marginal_std(t_continuous)
            return alpha_t * output + sigma_t * x
        elif self.model_type == "score":
            sigma_t = self.noise_schedule.marginal_std(t_continuous)
            return -sigma_t * output

    def noise_prediction_fn(self, x, t, quantized, conditioning, unconditional_guidance_scale, unconditional_conditioning):
        """
        Return the noise prediction model.
        """
        t_continuous = t.expand((x.shape[0]))

        if unconditional_guidance_scale == 1. or unconditional_conditioning is None:
            return self.noise_pred_fn(x, t_continuous, quantized, cond=conditioning)
        else:
            x_in = torch.cat([x] * 2)
            t_in = torch.cat([t_continuous] * 2)
            c_in = torch.cat([unconditional_conditioning, conditioning])
            noise_uncond, noise = self.noise_pred_fn(x_in, t_in, quantized, cond=c_in).chunk(2)
            return noise_uncond + unconditional_guidance_scale * (noise - noise_uncond)

    def data_prediction_fn(self, x, t, quantized, conditioning, unconditional_guidance_scale, unconditional_conditioning):
        """
        Return the data prediction model (with corrector).
        """
        noise = self.noise_prediction_fn(x, t, quantized, conditioning, unconditional_guidance_scale, unconditional_conditioning)
        alpha_t, sigma_t = self.noise_schedule.marginal_alpha(t), self.noise_schedule.marginal_std(t)
        x0 = (x - sigma_t * noise) / alpha_t
        if self.correcting_x0_fn is not None:
            x0 = self.correcting_x0_fn(x0, t)
        return x0

    def model_fn(self, x, t, quantized, conditioning, unconditional_guidance_scale, unconditional_conditioning):
        """
        Convert the model to the noise prediction model or the data prediction model. 
        """
        if self.algorithm_type == "dpmsolver++":
            return self.data_prediction_fn(x, t, quantized, conditioning, unconditional_guidance_scale, unconditional_conditioning)
        else:
            return self.noise_prediction_fn(x, t, quantized, conditioning, unconditional_guidance_scale, unconditional_conditioning)

    def get_time_steps(self, skip_type, t_T, t_0, N, device):
        """Compute the intermediate time steps for sampling.

        Args:
            skip_type: A `str`. The type for the spacing of the time steps. We support three types:
                - 'logSNR': uniform logSNR for the time steps.
                - 'time_uniform': uniform time for the time steps. (**Recommended for high-resolutional data**.)
                - 'time_quadratic': quadratic time for the time steps. (Used in DDIM for low-resolutional data.)
            t_T: A `float`. The starting time of the sampling (default is T).
            t_0: A `float`. The ending time of the sampling (default is epsilon).
            N: A `int`. The total number of the spacing of the time steps.
            device: A torch device.
        Returns:
            A pytorch tensor of the time steps, with the shape (N + 1,).
        """
        if skip_type == 'logSNR':
            lambda_T = self.noise_schedule.marginal_lambda(torch.tensor(t_T).to(device))
            lambda_0 = self.noise_schedule.marginal_lambda(torch.tensor(t_0).to(device))
            logSNR_steps = torch.linspace(lambda_T.cpu().item(), lambda_0.cpu().item(), N + 1).to(device)
            return self.noise_schedule.inverse_lambda(logSNR_steps)
        elif skip_type == 'time_uniform':
            return torch.linspace(t_T, t_0, N + 1).to(device)
        elif skip_type == 'time_quadratic':
            t_order = 2
            t = torch.linspace(t_T**(1. / t_order), t_0**(1. / t_order), N + 1).pow(t_order).to(device)
            return t
        else:
            raise ValueError("Unsupported skip_type {}, need to be 'logSNR' or 'time_uniform' or 'time_quadratic'".format(skip_type))

    def get_orders_and_timesteps_for_singlestep_solver(self, steps, order, skip_type, t_T, t_0, device):
        """
        Get the order of each step for sampling by the singlestep DPM-Solver.

        We combine both DPM-Solver-1,2,3 to use all the function evaluations, which is named as "DPM-Solver-fast".
        Given a fixed number of function evaluations by `steps`, the sampling procedure by DPM-Solver-fast is:
            - If order == 1:
                We take `steps` of DPM-Solver-1 (i.e. DDIM).
            - If order == 2:
                - Denote K = (steps // 2). We take K or (K + 1) intermediate time steps for sampling.
                - If steps % 2 == 0, we use K steps of DPM-Solver-2.
                - If steps % 2 == 1, we use K steps of DPM-Solver-2 and 1 step of DPM-Solver-1.
            - If order == 3:
                - Denote K = (steps // 3 + 1). We take K intermediate time steps for sampling.
                - If steps % 3 == 0, we use (K - 2) steps of DPM-Solver-3, and 1 step of DPM-Solver-2 and 1 step of DPM-Solver-1.
                - If steps % 3 == 1, we use (K - 1) steps of DPM-Solver-3 and 1 step of DPM-Solver-1.
                - If steps % 3 == 2, we use (K - 1) steps of DPM-Solver-3 and 1 step of DPM-Solver-2.

        ============================================
        Args:
            order: A `int`. The max order for the solver (2 or 3).
            steps: A `int`. The total number of function evaluations (NFE).
            skip_type: A `str`. The type for the spacing of the time steps. We support three types:
                - 'logSNR': uniform logSNR for the time steps.
                - 'time_uniform': uniform time for the time steps. (**Recommended for high-resolutional data**.)
                - 'time_quadratic': quadratic time for the time steps. (Used in DDIM for low-resolutional data.)
            t_T: A `float`. The starting time of the sampling (default is T).
            t_0: A `float`. The ending time of the sampling (default is epsilon).
            device: A torch device.
        Returns:
            orders: A list of the solver order of each step.
        """
        if order == 3:
            K = steps // 3 + 1
            if steps % 3 == 0:
                orders = [3,] * (K - 2) + [2, 1]
            elif steps % 3 == 1:
                orders = [3,] * (K - 1) + [1]
            else:
                orders = [3,] * (K - 1) + [2]
        elif order == 2:
            if steps % 2 == 0:
                K = steps // 2
                orders = [2,] * K
            else:
                K = steps // 2 + 1
                orders = [2,] * (K - 1) + [1]
        elif order == 1:
            K = 1
            orders = [1,] * steps
        else:
            raise ValueError("'order' must be '1' or '2' or '3'.")
        if skip_type == 'logSNR':
            # To reproduce the results in DPM-Solver paper
            timesteps_outer = self.get_time_steps(skip_type, t_T, t_0, K, device)
        else:
            timesteps_outer = self.get_time_steps(skip_type, t_T, t_0, steps, device)[torch.cumsum(torch.tensor([0,] + orders), 0).to(device)]
        return timesteps_outer, orders

    def denoise_to_zero_fn(self, x, s):
        """
        Denoise at the final step, which is equivalent to solve the ODE from lambda_s to infty by first-order discretization. 
        """
        return self.data_prediction_fn(x, s)

    def dpm_solver_first_update(self, x, s, t, quantized, model_s=None, conditioning=None, log_every_t=100, 
                                unconditional_guidance_scale=1., unconditional_conditioning=None):
        ns = self.noise_schedule
        dims = x.dim()
        lambda_s, lambda_t = ns.marginal_lambda(s), ns.marginal_lambda(t)
        h = lambda_t - lambda_s
        log_alpha_s, log_alpha_t = ns.marginal_log_mean_coeff(s), ns.marginal_log_mean_coeff(t)
        sigma_s, sigma_t = ns.marginal_std(s), ns.marginal_std(t)
        alpha_t = torch.exp(log_alpha_t)

        if self.algorithm_type == "dpmsolver++":
            phi_1 = torch.expm1(-h)
            if model_s is None:
                model_s = self.model_fn(x, s, quantized,
                                    conditioning=conditioning, 
                                    unconditional_guidance_scale=unconditional_guidance_scale, 
                                    unconditional_conditioning=unconditional_conditioning)
            x_t = (
                sigma_t / sigma_s * x
                - alpha_t * phi_1 * model_s
            )

            return x_t, model_s

        else:
            phi_1 = torch.expm1(h)
            if model_s is None:
                model_s = self.model_fn(x, s, quantized,
                                    conditioning=conditioning, 
                                    unconditional_guidance_scale=unconditional_guidance_scale, 
                                    unconditional_conditioning=unconditional_conditioning)
            x_t = (
                torch.exp(log_alpha_t - log_alpha_s) * x
                - (sigma_t * phi_1) * model_s
            )

            return x_t, model_s

    def singlestep_dpm_solver_second_update(self, x, s, t, quantized, r1=0.5, model_s=None, return_intermediate=False, solver_type='dpmsolver',
                                            conditioning=None, log_every_t=100, unconditional_guidance_scale=1., unconditional_conditioning=None):
        if solver_type not in ['dpmsolver', 'taylor']:
            raise ValueError("'solver_type' must be either 'dpmsolver' or 'taylor', got {}".format(solver_type))
        if r1 is None:
            r1 = 0.5
        ns = self.noise_schedule
        lambda_s, lambda_t = ns.marginal_lambda(s), ns.marginal_lambda(t)
        h = lambda_t - lambda_s
        lambda_s1 = lambda_s + r1 * h
        s1 = ns.inverse_lambda(lambda_s1)
        log_alpha_s, log_alpha_s1, log_alpha_t = ns.marginal_log_mean_coeff(s), ns.marginal_log_mean_coeff(s1), ns.marginal_log_mean_coeff(t)
        sigma_s, sigma_s1, sigma_t = ns.marginal_std(s), ns.marginal_std(s1), ns.marginal_std(t)
        alpha_s1, alpha_t = torch.exp(log_alpha_s1), torch.exp(log_alpha_t)

        if self.algorithm_type == "dpmsolver++":
            phi_11 = torch.expm1(-r1 * h)
            phi_1 = torch.expm1(-h)

            if model_s is None:
                model_s = self.model_fn(x, s, quantized, conditioning=conditioning, unconditional_guidance_scale=unconditional_guidance_scale, unconditional_conditioning=unconditional_conditioning)
            x_s1 = (
                (sigma_s1 / sigma_s) * x
                - (alpha_s1 * phi_11) * model_s
            )
            model_s1 = self.model_fn(x_s1, s1, quantized, conditioning=conditioning, unconditional_guidance_scale=unconditional_guidance_scale, unconditional_conditioning=unconditional_conditioning)
            if solver_type == 'dpmsolver':
                x_t = (
                    (sigma_t / sigma_s) * x
                    - (alpha_t * phi_1) * model_s
                    - (0.5 / r1) * (alpha_t * phi_1) * (model_s1 - model_s)
                )
            elif solver_type == 'taylor':
                x_t = (
                    (sigma_t / sigma_s) * x
                    - (alpha_t * phi_1) * model_s
                    + (1. / r1) * (alpha_t * (phi_1 / h + 1.)) * (model_s1 - model_s)
                )
        else:
            phi_11 = torch.expm1(r1 * h)
            phi_1 = torch.expm1(h)

            if model_s is None:
                model_s = self.model_fn(x, s, quantized, conditioning=conditioning, unconditional_guidance_scale=unconditional_guidance_scale, unconditional_conditioning=unconditional_conditioning)
            x_s1 = (
                torch.exp(log_alpha_s1 - log_alpha_s) * x
                - (sigma_s1 * phi_11) * model_s
            )
            model_s1 = self.model_fn(x_s1, s1, quantized, conditioning=conditioning, unconditional_guidance_scale=unconditional_guidance_scale, unconditional_conditioning=unconditional_conditioning)
            if solver_type == 'dpmsolver':
                x_t = (
                    torch.exp(log_alpha_t - log_alpha_s) * x
                    - (sigma_t * phi_1) * model_s
                    - (0.5 / r1) * (sigma_t * phi_1) * (model_s1 - model_s)
                )
            elif solver_type == 'taylor':
                x_t = (
                    torch.exp(log_alpha_t - log_alpha_s) * x
                    - (sigma_t * phi_1) * model_s
                    - (1. / r1) * (sigma_t * (phi_1 / h - 1.)) * (model_s1 - model_s)
                )

        return model_s, model_s1, x_s1, s1, x_t

    def singlestep_dpm_solver_third_update(self, x, s, t, r1=1./3., r2=2./3., model_s=None, model_s1=None, return_intermediate=False, solver_type='dpmsolver',
                                           conditioning=None, log_every_t=100, unconditional_guidance_scale=1., unconditional_conditioning=None):
        """
        Singlestep solver DPM-Solver-3 from time `s` to time `t`.

        Args:
            x: A pytorch tensor. The initial value at time `s`.
            s: A pytorch tensor. The starting time, with the shape (1,).
            t: A pytorch tensor. The ending time, with the shape (1,).
            r1: A `float`. The hyperparameter of the third-order solver.
            r2: A `float`. The hyperparameter of the third-order solver.
            model_s: A pytorch tensor. The model function evaluated at time `s`.
                If `model_s` is None, we evaluate the model by `x` and `s`; otherwise we directly use it.
            model_s1: A pytorch tensor. The model function evaluated at time `s1` (the intermediate time given by `r1`).
                If `model_s1` is None, we evaluate the model at `s1`; otherwise we directly use it.
            return_intermediate: A `bool`. If true, also return the model value at time `s`, `s1` and `s2` (the intermediate times).
            solver_type: either 'dpmsolver' or 'taylor'. The type for the high-order solvers.
                The type slightly impacts the performance. We recommend to use 'dpmsolver' type.
        Returns:
            x_t: A pytorch tensor. The approximated solution at time `t`.
        """
        if solver_type not in ['dpmsolver', 'taylor']:
            raise ValueError("'solver_type' must be either 'dpmsolver' or 'taylor', got {}".format(solver_type))
        if r1 is None:
            r1 = 1. / 3.
        if r2 is None:
            r2 = 2. / 3.
        ns = self.noise_schedule
        lambda_s, lambda_t = ns.marginal_lambda(s), ns.marginal_lambda(t)
        h = lambda_t - lambda_s
        lambda_s1 = lambda_s + r1 * h
        lambda_s2 = lambda_s + r2 * h
        s1 = ns.inverse_lambda(lambda_s1)
        s2 = ns.inverse_lambda(lambda_s2)
        log_alpha_s, log_alpha_s1, log_alpha_s2, log_alpha_t = ns.marginal_log_mean_coeff(s), ns.marginal_log_mean_coeff(s1), ns.marginal_log_mean_coeff(s2), ns.marginal_log_mean_coeff(t)
        sigma_s, sigma_s1, sigma_s2, sigma_t = ns.marginal_std(s), ns.marginal_std(s1), ns.marginal_std(s2), ns.marginal_std(t)
        alpha_s1, alpha_s2, alpha_t = torch.exp(log_alpha_s1), torch.exp(log_alpha_s2), torch.exp(log_alpha_t)

        if self.algorithm_type == "dpmsolver++":
            phi_11 = torch.expm1(-r1 * h)
            phi_12 = torch.expm1(-r2 * h)
            phi_1 = torch.expm1(-h)
            phi_22 = torch.expm1(-r2 * h) / (r2 * h) + 1.
            phi_2 = phi_1 / h + 1.
            phi_3 = phi_2 / h - 0.5

            if model_s is None:
                model_s = self.model_fn(x, s, conditioning=conditioning, 
                                        unconditional_guidance_scale=unconditional_guidance_scale, 
                                        unconditional_conditioning=unconditional_conditioning)
            if model_s1 is None:
                x_s1 = (
                    (sigma_s1 / sigma_s) * x
                    - (alpha_s1 * phi_11) * model_s
                )
                model_s1 = self.model_fn(x_s1, s1, conditioning=conditioning, 
                                        unconditional_guidance_scale=unconditional_guidance_scale, 
                                        unconditional_conditioning=unconditional_conditioning)
            x_s2 = (
                (sigma_s2 / sigma_s) * x
                - (alpha_s2 * phi_12) * model_s
                + r2 / r1 * (alpha_s2 * phi_22) * (model_s1 - model_s)
            )
            model_s2 = self.model_fn(x_s2, s2,
                                    conditioning=conditioning, 
                                    unconditional_guidance_scale=unconditional_guidance_scale, 
                                    unconditional_conditioning=unconditional_conditioning)
            if solver_type == 'dpmsolver':
                x_t = (
                    (sigma_t / sigma_s) * x
                    - (alpha_t * phi_1) * model_s
                    + (1. / r2) * (alpha_t * phi_2) * (model_s2 - model_s)
                )
            elif solver_type == 'taylor':
                D1_0 = (1. / r1) * (model_s1 - model_s)
                D1_1 = (1. / r2) * (model_s2 - model_s)
                D1 = (r2 * D1_0 - r1 * D1_1) / (r2 - r1)
                D2 = 2. * (D1_1 - D1_0) / (r2 - r1)
                x_t = (
                    (sigma_t / sigma_s) * x
                    - (alpha_t * phi_1) * model_s
                    + (alpha_t * phi_2) * D1
                    - (alpha_t * phi_3) * D2
                )
        else:
            phi_11 = torch.expm1(r1 * h)
            phi_12 = torch.expm1(r2 * h)
            phi_1 = torch.expm1(h)
            phi_22 = torch.expm1(r2 * h) / (r2 * h) - 1.
            phi_2 = phi_1 / h - 1.
            phi_3 = phi_2 / h - 0.5

            if model_s is None:
                model_s = self.model_fn(x, s,
                                        conditioning=conditioning, 
                                        unconditional_guidance_scale=unconditional_guidance_scale, 
                                        unconditional_conditioning=unconditional_conditioning)
            if model_s1 is None:
                x_s1 = (
                    (torch.exp(log_alpha_s1 - log_alpha_s)) * x
                    - (sigma_s1 * phi_11) * model_s
                )
                model_s1 = self.model_fn(x_s1, s1,
                                        conditioning=conditioning, 
                                        unconditional_guidance_scale=unconditional_guidance_scale, 
                                        unconditional_conditioning=unconditional_conditioning)
            x_s2 = (
                (torch.exp(log_alpha_s2 - log_alpha_s)) * x
                - (sigma_s2 * phi_12) * model_s
                - r2 / r1 * (sigma_s2 * phi_22) * (model_s1 - model_s)
            )
            model_s2 = self.model_fn(x_s2, s2,
                                    conditioning=conditioning, 
                                    unconditional_guidance_scale=unconditional_guidance_scale, 
                                    unconditional_conditioning=unconditional_conditioning)
            if solver_type == 'dpmsolver':
                x_t = (
                    (torch.exp(log_alpha_t - log_alpha_s)) * x
                    - (sigma_t * phi_1) * model_s
                    - (1. / r2) * (sigma_t * phi_2) * (model_s2 - model_s)
                )
            elif solver_type == 'taylor':
                D1_0 = (1. / r1) * (model_s1 - model_s)
                D1_1 = (1. / r2) * (model_s2 - model_s)
                D1 = (r2 * D1_0 - r1 * D1_1) / (r2 - r1)
                D2 = 2. * (D1_1 - D1_0) / (r2 - r1)
                x_t = (
                    (torch.exp(log_alpha_t - log_alpha_s)) * x
                    - (sigma_t * phi_1) * model_s
                    - (sigma_t * phi_2) * D1
                    - (sigma_t * phi_3) * D2
                )

        if return_intermediate:
            return x_t, {'model_s': model_s, 'model_s1': model_s1, 'model_s2': model_s2}
        else:
            return x_t

    def singlestep_dpm_solver_update(self, x, s, t, order, quantized, solver_type='dpmsolver', r1=None, r2=None,  
                                    conditioning=None, log_every_t=100, unconditional_guidance_scale=1., 
                                    unconditional_conditioning=None, return_intermediate=False):
        """
        Singlestep DPM-Solver with the order `order` from time `s` to time `t`.

        Args:
            x: A pytorch tensor. The initial value at time `s`.
            s: A pytorch tensor. The starting time, with the shape (1,).
            t: A pytorch tensor. The ending time, with the shape (1,).
            order: A `int`. The order of DPM-Solver. We only support order == 1 or 2 or 3.
            return_intermediate: A `bool`. If true, also return the model value at time `s`, `s1` and `s2` (the intermediate times).
            solver_type: either 'dpmsolver' or 'taylor'. The type for the high-order solvers.
                The type slightly impacts the performance. We recommend to use 'dpmsolver' type.
            r1: A `float`. The hyperparameter of the second-order or third-order solver.
            r2: A `float`. The hyperparameter of the third-order solver.
        Returns:
            x_t: A pytorch tensor. The approximated solution at time `t`.
        """
        if order == 1:
            return self.dpm_solver_first_update(x, s, t, quantized, conditioning=conditioning, log_every_t=log_every_t, 
                                                unconditional_guidance_scale=unconditional_guidance_scale, 
                                                unconditional_conditioning=unconditional_conditioning)
        elif order == 2:
            return self.singlestep_dpm_solver_second_update(x, s, t, quantized, return_intermediate=return_intermediate, 
                                                            solver_type=solver_type, r1=r1,
                                                            conditioning=conditioning, log_every_t=log_every_t, 
                                                            unconditional_guidance_scale=unconditional_guidance_scale, 
                                                            unconditional_conditioning=unconditional_conditioning)
        elif order == 3:
            return self.singlestep_dpm_solver_third_update(x, s, t, return_intermediate=return_intermediate, 
                                                            solver_type=solver_type, r1=r1, r2=r2,
                                                            conditioning=conditioning, log_every_t=log_every_t, 
                                                            unconditional_guidance_scale=unconditional_guidance_scale, 
                                                            unconditional_conditioning=unconditional_conditioning)
        else:
            raise ValueError("Solver order must be 1 or 2 or 3, got {}".format(order))

    def dpm_solver_adaptive(self, x, order, t_T, t_0, h_init=0.05, atol=0.0078, rtol=0.05, theta=0.9, t_err=1e-5, solver_type='dpmsolver'):
        """
        The adaptive step size solver based on singlestep DPM-Solver.

        Args:
            x: A pytorch tensor. The initial value at time `t_T`.
            order: A `int`. The (higher) order of the solver. We only support order == 2 or 3.
            t_T: A `float`. The starting time of the sampling (default is T).
            t_0: A `float`. The ending time of the sampling (default is epsilon).
            h_init: A `float`. The initial step size (for logSNR).
            atol: A `float`. The absolute tolerance of the solver. For image data, the default setting is 0.0078, followed [1].
            rtol: A `float`. The relative tolerance of the solver. The default setting is 0.05.
            theta: A `float`. The safety hyperparameter for adapting the step size. The default setting is 0.9, followed [1].
            t_err: A `float`. The tolerance for the time. We solve the diffusion ODE until the absolute error between the 
                current time and `t_0` is less than `t_err`. The default setting is 1e-5.
            solver_type: either 'dpmsolver' or 'taylor'. The type for the high-order solvers.
                The type slightly impacts the performance. We recommend to use 'dpmsolver' type.
        Returns:
            x_0: A pytorch tensor. The approximated solution at time `t_0`.

        [1] A. Jolicoeur-Martineau, K. Li, R. Piché-Taillefer, T. Kachman, and I. Mitliagkas, "Gotta go fast when generating data with score-based models," arXiv preprint arXiv:2105.14080, 2021.
        """
        ns = self.noise_schedule
        s = t_T * torch.ones((1,)).to(x)
        lambda_s = ns.marginal_lambda(s)
        lambda_0 = ns.marginal_lambda(t_0 * torch.ones_like(s).to(x))
        h = h_init * torch.ones_like(s).to(x)
        x_prev = x
        nfe = 0
        if order == 2:
            r1 = 0.5
            lower_update = lambda x, s, t: self.dpm_solver_first_update(x, s, t, return_intermediate=True)
            higher_update = lambda x, s, t, **kwargs: self.singlestep_dpm_solver_second_update(x, s, t, r1=r1, solver_type=solver_type, **kwargs)
        elif order == 3:
            r1, r2 = 1. / 3., 2. / 3.
            lower_update = lambda x, s, t: self.singlestep_dpm_solver_second_update(x, s, t, r1=r1, return_intermediate=True, solver_type=solver_type)
            higher_update = lambda x, s, t, **kwargs: self.singlestep_dpm_solver_third_update(x, s, t, r1=r1, r2=r2, solver_type=solver_type, **kwargs)
        else:
            raise ValueError("For adaptive step size solver, order must be 2 or 3, got {}".format(order))
        while torch.abs((s - t_0)).mean() > t_err:
            t = ns.inverse_lambda(lambda_s + h)
            x_lower, lower_noise_kwargs = lower_update(x, s, t)
            x_higher = higher_update(x, s, t, **lower_noise_kwargs)
            delta = torch.max(torch.ones_like(x).to(x) * atol, rtol * torch.max(torch.abs(x_lower), torch.abs(x_prev)))
            norm_fn = lambda v: torch.sqrt(torch.square(v.reshape((v.shape[0], -1))).mean(dim=-1, keepdim=True))
            E = norm_fn((x_higher - x_lower) / delta).max()
            if torch.all(E <= 1.):
                x = x_higher
                s = t
                x_prev = x_lower
                lambda_s = ns.marginal_lambda(s)
            h = torch.min(theta * h * torch.float_power(E, -1. / order).float(), lambda_0 - lambda_s)
            nfe += order
        print('adaptive solver nfe', nfe)
        return x

    def add_noise(self, x, t, noise=None):
        """
        Compute the noised input xt = alpha_t * x + sigma_t * noise. 

        Args:
            x: A `torch.Tensor` with shape `(batch_size, *shape)`.
            t: A `torch.Tensor` with shape `(t_size,)`.
        Returns:
            xt with shape `(t_size, batch_size, *shape)`.
        """
        alpha_t, sigma_t = self.noise_schedule.marginal_alpha(t), self.noise_schedule.marginal_std(t)
        if noise is None:
            noise = torch.randn((t.shape[0], *x.shape), device=x.device)
        x = x.reshape((-1, *x.shape))
        xt = expand_dims(alpha_t, x.dim()) * x + expand_dims(sigma_t, x.dim()) * noise
        if t.shape[0] == 1:
            return xt.squeeze(0)
        else:
            return xt

    def inverse(self, x, steps=20, t_start=None, t_end=None, order=2, skip_type='time_uniform',
        method='multistep', lower_order_final=True, denoise_to_zero=False, solver_type='dpmsolver',
        atol=0.0078, rtol=0.05, return_intermediate=False,
    ):
        """
        Inverse the sample `x` from time `t_start` to `t_end` by DPM-Solver.
        For discrete-time DPMs, we use `t_start=1/N`, where `N` is the total time steps during training.
        """
        t_0 = 1. / self.noise_schedule.total_N if t_start is None else t_start
        t_T = self.noise_schedule.T if t_end is None else t_end
        assert t_0 > 0 and t_T > 0, "Time range needs to be greater than 0. For discrete-time DPMs, it needs to be in [1 / N, 1], where N is the length of betas array"
        return self.sample(x, steps=steps, t_start=t_0, t_end=t_T, order=order, skip_type=skip_type,
            method=method, lower_order_final=lower_order_final, denoise_to_zero=denoise_to_zero, solver_type=solver_type,
            atol=atol, rtol=rtol, return_intermediate=return_intermediate)

    def sample_loop(self, x, conditioning, fp_steps=20, quant_steps=40, t_start=None, t_end=None, skip_type='time_uniform',
        method='singlestep', lower_order_final=True, denoise_to_zero=False, solver_type='dpmsolver',
        atol=0.0078, rtol=0.05, log_every_t=100, unconditional_guidance_scale=1., unconditional_conditioning=None,
    ):
        t_0 = 1. / self.noise_schedule.total_N if t_end is None else t_end
        t_T = self.noise_schedule.T if t_start is None else t_start
        assert t_0 > 0 and t_T > 0, "Time range needs to be greater than 0. For discrete-time DPMs, it needs to be in [1 / N, 1], where N is the length of betas array"
        if self.correcting_xt_fn is not None:
            assert method in ['multistep', 'singlestep', 'singlestep_fixed'], "Cannot use adaptive solver when correcting_xt_fn is not None"
        device = x.device

        if method in ['singlestep', 'singlestep_fixed']:
            if method == 'singlestep':
                timesteps_outer_fp, orders_fp = self.get_orders_and_timesteps_for_singlestep_solver(steps=fp_steps, order=self.fp_order, skip_type=skip_type, t_T=t_T, t_0=t_0, device=device)
                timesteps_outer_quant, orders_quant = self.get_orders_and_timesteps_for_singlestep_solver(steps=quant_steps, order=self.quant_order, skip_type=skip_type, t_T=t_T, t_0=t_0, device=device)
            elif method == 'singlestep_fixed':
                K = fp_steps // self.fp_order
                orders_fp = [self.fp_order,] * K
                timesteps_outer_fp = self.get_time_steps(skip_type=skip_type, t_T=t_T, t_0=t_0, N=K, device=device)
            for step, order_fp in tqdm(enumerate(orders_fp), total=len(orders_fp), desc="Processing steps"):
                # FP32 model params
                s_fp, t_fp = timesteps_outer_fp[step], timesteps_outer_fp[step + 1]
                timesteps_inner_fp = self.get_time_steps(skip_type=skip_type, t_T=s_fp.item(), t_0=t_fp.item(), N=order_fp, device=device)
                lambda_inner_fp = self.noise_schedule.marginal_lambda(timesteps_inner_fp)
                h = lambda_inner_fp[-1] - lambda_inner_fp[0]
                r1_fp = None if order_fp <= 1 else (lambda_inner_fp[1] - lambda_inner_fp[0]) / h
                r2_fp = None if order_fp <= 2 else (lambda_inner_fp[2] - lambda_inner_fp[0]) / h
                
                # quant model params
                s_q, t_q = timesteps_outer_quant[step], timesteps_outer_quant[step + 1]
                timesteps_inner_quant = self.get_time_steps(skip_type=skip_type, t_T=s_q.item(), t_0=t_q.item(), N=self.quant_order, device=device)
                lambda_inner_quant = self.noise_schedule.marginal_lambda(timesteps_inner_quant)
                h = lambda_inner_quant[-1] - lambda_inner_quant[0]
                r1_q = None if self.quant_order <= 1 else (lambda_inner_quant[1] - lambda_inner_quant[0]) / h
                r2_q = None if self.quant_order <= 2 else (lambda_inner_quant[2] - lambda_inner_quant[0]) / h
    
                x = self.p_dpm_solver_sample(x, s_fp=s_fp, t_fp=t_fp, solver_type=solver_type, r1_fp=r1_fp, r2_fp=r2_fp, 
                                            s_q=s_q, t_q=t_q, r1_q=r1_q, r2_q=r2_q, conditioning=conditioning, 
                                            log_every_t=log_every_t, unconditional_guidance_scale=unconditional_guidance_scale, 
                                            unconditional_conditioning=unconditional_conditioning)

                if self.correcting_xt_fn is not None:
                    x = self.correcting_xt_fn(x, t, step)
        else:
            raise ValueError("Got wrong method {}".format(method))
        if denoise_to_zero:
            t = torch.ones((1,)).to(device) * t_0
            x = self.denoise_to_zero_fn(x, t)
            if self.correcting_xt_fn is not None:
                x = self.correcting_xt_fn(x, t, step + 1)

        return x, timesteps_outer_fp

    def p_dpm_solver_sample(self, x, s_fp, t_fp, solver_type, r1_fp, r2_fp, s_q, t_q, 
                            r1_q, r2_q, conditioning, log_every_t, unconditional_guidance_scale, unconditional_conditioning):
        self.optimizer.zero_grad()

        x_fp = None
        if self.fp_order == self.quant_order:
            if self.fp_order == 1:
                x_fp, model_s_fp = self.singlestep_dpm_solver_update(x, s_fp, t_fp, self.fp_order, quantized=False, solver_type=solver_type, r1=r1_fp, r2=r2_fp, 
                                                                    conditioning=conditioning, log_every_t=log_every_t,
                                                                    unconditional_guidance_scale=unconditional_guidance_scale, 
                                                                    unconditional_conditioning=unconditional_conditioning)
                
                x_quant, model_s_quant = self.singlestep_dpm_solver_update(x, s_q, t_q, self.quant_order, quantized=True, solver_type=solver_type, r1=r1_q, r2=r2_q, 
                                                                    conditioning=conditioning, log_every_t=log_every_t,
                                                                    unconditional_guidance_scale=unconditional_guidance_scale, 
                                                                    unconditional_conditioning=unconditional_conditioning)

                loss = F.mse_loss(model_s_quant, model_s_fp, size_average=False)
                # model_s_quant_flat = model_s_quant.view(model_s_quant.size(0), -1)  # Shape: [batch_size, 3*64*64]
                # model_s_fp_flat = model_s_fp.view(model_s_fp.size(0), -1)  # Shape: [batch_size, 3*64*64]
                # cosine_similarity = F.cosine_similarity(model_s_quant_flat, model_s_fp_flat)
                # cos_loss = torch.sum(1 - cosine_similarity)
                # print('mse loss: ', mse_loss.item(), " cos loss: ", cos_loss)
                # loss = mse_loss + 100. * cos_loss
                loss.backward()
                self.optimizer.step()
                self.lr_scheduler.step()
                print('loss: ', loss.item())
            elif self.fp_order ==  2:
                model_s_q, model_s1_q, x_s1_q, s1_q, x_quant = self.singlestep_dpm_solver_update(x, s_q, t_q, self.quant_order, quantized=True, solver_type=solver_type, r1=r1_q, r2=r2_q, 
                                                            conditioning=conditioning, log_every_t=log_every_t,
                                                            unconditional_guidance_scale=unconditional_guidance_scale, 
                                                            unconditional_conditioning=unconditional_conditioning)
                
                model_s_fp, model_s1_fp, x_s1_fp, s1_fp, x_fp = self.singlestep_dpm_solver_update(x, s_fp, t_fp, self.fp_order, quantized=False, solver_type=solver_type, r1=r1_fp, r2=r2_fp, 
                                                            conditioning=conditioning, log_every_t=log_every_t,
                                                            unconditional_guidance_scale=unconditional_guidance_scale, 
                                                            unconditional_conditioning=unconditional_conditioning)

                loss = F.mse_loss(model_s_q, model_s_fp, size_average=False) + F.mse_loss(model_s1_q, model_s1_fp, size_average=False)
                loss.backward()
                self.optimizer.step()
                self.lr_scheduler.step()
                print('loss: ', loss.item())
        elif self.fp_order != self.quant_order:
            if self.fp_order == 2 and self.quant_order == 1:
                model_s_fp, model_s1, x_s1, s1, x_fp = self.singlestep_dpm_solver_update(x, s_fp, t_fp, self.fp_order, quantized=False, solver_type=solver_type, r1=r1_fp, r2=r2_fp, 
                                                            conditioning=conditioning, log_every_t=log_every_t,
                                                            unconditional_guidance_scale=unconditional_guidance_scale, 
                                                            unconditional_conditioning=unconditional_conditioning)

                x_quant, model_s_quant = self.singlestep_dpm_solver_update(x, s_q, t_q, self.quant_order, quantized=True, solver_type=solver_type, r1=r1_q, r2=r2_q, 
                                                                    conditioning=conditioning, log_every_t=log_every_t,
                                                                    unconditional_guidance_scale=unconditional_guidance_scale, 
                                                                    unconditional_conditioning=unconditional_conditioning)

                model_s_quant_flat = model_s_quant.view(model_s_quant.size(0), -1)  # Shape: [batch_size, 3*64*64]
                model_s1_flat = model_s1.view(model_s1.size(0), -1)  # Shape: [batch_size, 3*64*64]

                # mse_loss = F.mse_loss(model_s_quant, model_s1, reduction='none').sum(dim=(1, 2, 3)).mean()
            
                mse_loss = F.mse_loss(model_s_quant, model_s1, size_average=False)
                # Calculate cosine similarity along the last dimension (now only two dimensions remain)
                cosine_similarity = F.cosine_similarity(model_s_quant_flat, model_s1_flat)
                cos_loss = torch.sum(1 - cosine_similarity)
                print('mse loss: ', mse_loss.item(), " cos loss: ", cos_loss)
                loss = mse_loss + 100. * cos_loss
                loss.backward()
                self.optimizer.step()
                self.lr_scheduler.step()
                print('loss: ', loss.item())

        return x_fp

    def sample(self, batch_size, fp_steps, quant_steps, shape, skip_type='quad', conditioning=None, method="singlestep",
            verbose=True, x_T=None, log_every_t=100, unconditional_guidance_scale=1., unconditional_conditioning=None, **kwargs ):
        if conditioning is not None:
            if isinstance(conditioning, dict):
                cbs = conditioning[list(conditioning.keys())[0]].shape[0]
                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
            else:
                if conditioning.shape[0] != batch_size:
                    print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

        # sampling
        C, H, W = shape
        shape = (batch_size, C, H, W)

        device = self.fp_model.betas.device

        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T

        img, timesteps_outer_fp = self.sample_loop(
            img,
            conditioning,
            fp_steps=fp_steps, 
            quant_steps=quant_steps,
            skip_type="time_quadratic" if skip_type == 'quad' else 'time_uniform',
            method=method,
            log_every_t=log_every_t,
            unconditional_guidance_scale=unconditional_guidance_scale,
            unconditional_conditioning=unconditional_conditioning
        )
        return img, timesteps_outer_fp

