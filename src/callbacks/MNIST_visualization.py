from __future__ import annotations

from typing import Tuple
from pathlib import Path

import torch
from lightning import Callback
import matplotlib.pyplot as plt


class MNISTPredictionVisualizationCallback(Callback):
	"""Log a simple figure with 3 MNIST images and predictions during validation.

	This callback grabs the first validation batch each epoch, takes the first
	``num_images`` samples, runs a forward pass, and logs a matplotlib figure
	showing the images with their predicted and true labels.

	Notes
	-----
	- Designed to work out of the box with TensorBoard logger
	  (``trainer.logger.experiment.add_figure``).
	- Kept intentionally minimal and framework-agnostic otherwise.
	- Only logs on the global zero process in distributed runs.
	"""

	def __init__(self, num_images: int = 3) -> None:
		super().__init__()
		self.num_images = max(1, num_images)
		self._logged_this_epoch = False

	# --- Epoch lifecycle -------------------------------------------------
	def on_validation_epoch_start(self, trainer, pl_module) -> None:  # type: ignore[override]
		self._logged_this_epoch = False

	# --- Main hook -------------------------------------------------------
	def on_validation_batch_end(  # type: ignore[override]
		self,
		trainer,
		pl_module,
		outputs,
		batch: Tuple[torch.Tensor, torch.Tensor],
		batch_idx: int,
		dataloader_idx: int = 0,
	) -> None:
		# Log only once per epoch and only from rank 0
		if self._logged_this_epoch or not getattr(trainer, "is_global_zero", True):
			return

		x, y = batch
		if x.ndim != 4:  # expected [B, C, H, W]
			return

		x = x[: self.num_images]
		y = y[: self.num_images]

		# Ensure tensors are on the model's device for forward
		x = x.to(pl_module.device)

		# Forward pass (no grad, keep current eval/train mode as set by Lightning)
		with torch.no_grad():
			logits = pl_module(x)
			preds = torch.argmax(logits, dim=1)

		# Denormalize for visualization if MNIST normalize(mean=0.1307, std=0.3081) was used
		imgs = self._denormalize_mnist(x)

		# Build a simple 1xN matplotlib figure (grayscale)
		fig, axes = plt.subplots(1, imgs.shape[0], figsize=(3 * imgs.shape[0], 3))
		if imgs.shape[0] == 1:
			axes = [axes]  # make iterable

		for i, ax in enumerate(axes):
			img = imgs[i].detach().cpu().squeeze(0)  # [H, W]
			ax.imshow(img, cmap="gray")
			ax.axis("off")
			ax.set_title(f"pred: {int(preds[i])} | true: {int(y[i])}")

		fig.tight_layout()

		# Try to log via TensorBoard-like API if available
		logger = getattr(trainer, "logger", None)
		experiment = getattr(logger, "experiment", None)
		if experiment is not None and hasattr(experiment, "add_figure"):
			experiment.add_figure("MNIST/predictions", fig, global_step=trainer.global_step)
		else:
			# Fallback: save the figure to disk
			root = getattr(trainer, "default_root_dir", ".")
			out_dir = Path(root) / "visualizations"
			out_dir.mkdir(parents=True, exist_ok=True)
			fname = f"mnist_predictions_epoch{trainer.current_epoch:03d}_step{trainer.global_step:06d}.png"
			fig.savefig(out_dir / fname)

		# Close the figure to free memory regardless of logging backend
		plt.close(fig)

		self._logged_this_epoch = True

	# --- Helpers ---------------------------------------------------------
	@staticmethod
	def _denormalize_mnist(x: torch.Tensor) -> torch.Tensor:
		"""Denormalize MNIST tensors that were normalized with (mean=0.1307, std=0.3081).

		Expects input in shape [B, 1, H, W]. Returns clamped values in [0, 1].
		If the input wasn't normalized, this still produces reasonable output.
		"""
		mean = 0.1307
		std = 0.3081
		# x' = x * std + mean
		out = x * std + mean
		return out.clamp(0.0, 1.0)


__all__ = ["MNISTPredictionVisualizationCallback"]

