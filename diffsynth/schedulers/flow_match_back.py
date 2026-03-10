import torch
import numpy as np


class FlowMatchScheduler:

    def __init__(
        self,
        num_inference_steps=100,
        num_train_timesteps=1000,
        shift=3.0,
        sigma_max=1.0,
        sigma_min=0.003 / 1.002,
        inverse_timesteps=False,
        extra_one_step=False,
        reverse_sigmas=False,
    ):
        self.num_train_timesteps = num_train_timesteps
        self.shift = shift
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.inverse_timesteps = inverse_timesteps
        self.extra_one_step = extra_one_step
        self.reverse_sigmas = reverse_sigmas
        self.training_weight_type = "default"
        self.set_timesteps(num_inference_steps)

    def set_training_weight(self, training_weight_type):
        assert training_weight_type in [
            "default",
            "equal",
            "early",
            "late",
        ], "training_weight_type must be one of 'default', 'equal', 'early', or 'late'"
        self.training_weight_type = training_weight_type

    def set_timesteps(
        self,
        schedule_mode="default",
        num_inference_steps=100,
        denoising_strength=1.0,
        training=False,
        shift=None,
    ):
        if shift is not None:
            self.shift = shift
        sigma_start = (
            self.sigma_min + (self.sigma_max -
                              self.sigma_min) * denoising_strength
        )
        if self.extra_one_step:
            self.sigmas = torch.linspace(
                sigma_start, self.sigma_min, num_inference_steps + 1
            )[:-1]
        else:
            self.sigmas = torch.linspace(
                sigma_start, self.sigma_min, num_inference_steps
            )

        if self.inverse_timesteps:
            self.sigmas = torch.flip(self.sigmas, dims=[0])
        if schedule_mode == "default":
            self.sigmas = (
                self.shift * self.sigmas / (1 + (self.shift - 1) * self.sigmas)
            )
        elif schedule_mode == "cosine":

            def cosine_sigma_schedule(T, s=0.008):
                steps = T + 1
                t = np.linspace(0, T, steps) / T
                alphas_cumprod = np.cos((t + s) / (1 + s) * np.pi / 2) ** 2
                alphas_cumprod = alphas_cumprod / \
                    alphas_cumprod[0]  # normalize
                betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
                betas = np.clip(betas, 1e-8, 0.999)
                # convert betas to sigmas
                sigmas = np.sqrt(betas)
                sigmas = sigma_start * sigmas / sigmas.max()  # scale to sigma_start
                return torch.tensor(sigmas, dtype=torch.float32)

            self.sigmas = cosine_sigma_schedule(num_inference_steps)
            self.sigmas = torch.flip(self.sigmas, dims=[0])

        elif schedule_mode == "linear":
            self.sigmas = torch.linspace(
                sigma_start, self.sigma_min, num_inference_steps
            )

        if self.reverse_sigmas:
            self.sigmas = 1 - self.sigmas

        self.timesteps = self.sigmas * self.num_train_timesteps

        if training:
            x = self.timesteps
            y = torch.exp(
                -2 * ((x - num_inference_steps / 2) / num_inference_steps) ** 2
            )
            y_shifted = y - y.min()
            bsmntw_weighing = y_shifted * \
                (num_inference_steps / y_shifted.sum())

            if self.training_weight_type == "equal":
                bsmntw_weighing = (
                    torch.ones_like(bsmntw_weighing) * 1.795
                )  # 1.795 is the previous largest weight
            elif self.training_weight_type == "late":
                bsmntw_weighing = 1.795 - 1 * (
                    x / torch.max(x)
                )  # Low when it is early timestep, high when late timestep because less denoising is needed
            else:
                pass  # because the default is already set as above

            self.linear_timesteps_weights = bsmntw_weighing
            self.training = True
        else:
            self.training = False

    def step(self, model_output, timestep, sample, to_final=False, **kwargs):
        # if isinstance(timestep, torch.Tensor):
        #     timestep = timestep.cpu()
        # timestep_id = torch.argmin((self.timesteps - timestep).abs())
        # sigma = self.sigmas[timestep_id]
        # if to_final or timestep_id + 1 >= len(self.timesteps):
        #     sigma_ = 1 if (
        #         self.inverse_timesteps or self.reverse_sigmas) else 0
        # else:
        #     sigma_ = self.sigmas[timestep_id + 1]
        # TODO
        return sample - model_output
        prev_sample = sample + model_output * (sigma_ - sigma)
        return prev_sample

    def return_to_timestep(self, timestep, sample, sample_stablized):
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.cpu()
        timestep_id = torch.argmin((self.timesteps - timestep).abs())
        sigma = self.sigmas[timestep_id]
        model_output = (sample - sample_stablized) / sigma
        return model_output

    def add_noise(self, original_samples, noise, timestep):
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.cpu()
        timestep_id = torch.argmin((self.timesteps - timestep).abs())
        sigma = self.sigmas[timestep_id]
        # print(f"Sample shape: {original_samples.shape}, Noise shape: {noise.shape}, Sigma: {sigma}")
        # sample = (1 - sigma) * original_samples + sigma * noise
        sample = original_samples + noise
        return noise

    def training_target(self, sample, noise, timestep):
        target = noise - sample
        return target

    def training_weight(self, timestep):
        timestep_id = torch.argmin(
            (self.timesteps - timestep.to(self.timesteps.device)).abs()
        )
        weights = self.linear_timesteps_weights[timestep_id]
        return weights


if __name__ == "__main__":
    scheduler = FlowMatchScheduler()
    scheduler.set_training_weight("default")
    scheduler.set_timesteps(
        num_inference_steps=1, training=True, schedule_mode="default"
    )
    print("Timesteps:", scheduler.timesteps)
    print("Sigmas:", scheduler.sigmas)
    print(f"Training weights: {scheduler.linear_timesteps_weights}")
    for step, sigma, weight in zip(scheduler.timesteps, scheduler.sigmas, scheduler.linear_timesteps_weights):
        print(f"Step: {step}, Sigma: {sigma}, Weight: {weight}")
