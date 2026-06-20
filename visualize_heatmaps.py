from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchcam.methods import LayerCAM


@dataclass
class HeatmapResult:
	image: np.ndarray
	heatmap: np.ndarray
	overlay: np.ndarray
	score: float


class _CrossViewSimilarityWrapper(nn.Module):
	"""
	Minimal wrapper that scores one image against a pre-computed reference
	embedding.  Accepting the reference embedding (rather than the raw image)
	ensures the LayerCAM forward hook ONLY fires during the single forward pass
	of the image being explained, where requires_grad=True is guaranteed.
	"""

	def __init__(self, base_model: nn.Module, use_logit_scale: bool = True):
		super().__init__()
		self.base_model = base_model
		self.use_logit_scale = use_logit_scale

	def _encode(self, x: torch.Tensor) -> torch.Tensor:
		out = self.base_model(x)
		if isinstance(out, tuple):
			out = out[0]
		return out

	def _scale(self) -> torch.Tensor:
		if self.use_logit_scale and hasattr(self.base_model, "logit_scale"):
			return self.base_model.logit_scale.exp()
		return torch.tensor(1.0, device=next(self.base_model.parameters()).device)

	def forward(self, explained_img: torch.Tensor, ref_emb: torch.Tensor) -> torch.Tensor:
		"""
		Args:
			explained_img: image being explained; must have requires_grad=True.
			ref_emb:       pre-computed embedding of the reference image (no grad).
		Returns:
			Similarity score of shape [B, 1] for LayerCAM.
		"""
		emb = self._encode(explained_img)
		sim = F.cosine_similarity(emb, ref_emb, dim=-1) * self._scale()
		return sim.unsqueeze(1)


class CrossViewLayerCAMVisualizer:
	"""
	LayerCAM visualizer for cross-view geolocalization models (Sample4Geo).

	Usage:
		visualizer = CrossViewLayerCAMVisualizer(model)
		sat_result = visualizer.generate_heatmap(sat_tensor, grd_tensor, explain_view="satellite")
		grd_result = visualizer.generate_heatmap(sat_tensor, grd_tensor, explain_view="ground")

	Input tensors are expected to be normalized in the same way as training/eval.
	"""

	def __init__(
		self,
		model: nn.Module,
		target_layer: Optional[nn.Module] = None,
		mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
		std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
		device: Optional[torch.device] = None,
		use_logit_scale: bool = True,
	):
		self.model = model
		self.model.eval()

		if device is None:
			device = next(model.parameters()).device
		self.device = device

		self.mean = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
		self.std = np.array(std, dtype=np.float32).reshape(1, 1, 3)

		self.target_layer = target_layer if target_layer is not None else self._find_default_target_layer(model)
		self.use_logit_scale = use_logit_scale



	def _find_default_target_layer(self, model: nn.Module) -> nn.Module:
		# Preferred ConvNeXt layer used in this codebase.
		if hasattr(model, "model") and hasattr(model.model, "stages"):
			stages = model.model.stages
			if len(stages) > 0 and hasattr(stages[-1], "blocks") and len(stages[-1].blocks) > 0:
				block = stages[-1].blocks[-1]
				if hasattr(block, "conv_dw"):
					return block.conv_dw

		# Fallback: use the last convolutional layer in the model.
		for module in reversed(list(model.modules())):
			if isinstance(module, nn.Conv2d):
				return module

		raise RuntimeError(
			"Could not infer a convolutional target layer for LayerCAM. "
			"Please pass target_layer explicitly."
		)

	def _encode_single(self, img: torch.Tensor) -> torch.Tensor:
		"""Encode one image under no_grad, returning a detached embedding."""
		with torch.no_grad():
			out = self.model(img)
			if isinstance(out, tuple):
				out = out[0]
			return out.detach()

	def _to_uint8_rgb(self, normalized_tensor: torch.Tensor) -> np.ndarray:
		img = normalized_tensor.detach().cpu().permute(1, 2, 0).numpy()
		img = img * self.std + self.mean
		img = np.clip(img, 0.0, 1.0)
		return (img * 255.0).astype(np.uint8)

	@staticmethod
	def _normalize_map(cam_map: np.ndarray) -> np.ndarray:
		cam_map = cam_map.astype(np.float32)
		min_v = float(cam_map.min())
		max_v = float(cam_map.max())
		if max_v <= min_v:
			return np.zeros_like(cam_map, dtype=np.float32)
		return (cam_map - min_v) / (max_v - min_v)

	def generate_heatmap(
		self,
		satellite: torch.Tensor,
		ground: torch.Tensor,
		explain_view: str = "satellite",
		alpha: float = 0.45,
	) -> HeatmapResult:
		"""
		Generate LayerCAM visualization for either satellite or ground image.

		Args:
			satellite: Tensor [B, 3, H, W], normalized.
			ground: Tensor [B, 3, H, W], normalized.
			explain_view: "satellite" or "ground".
			alpha: blending factor for overlay.
		"""
		if satellite.dim() != 4 or ground.dim() != 4:
			raise ValueError("satellite and ground must be 4D tensors [B, C, H, W].")
		if satellite.shape[0] != 1 or ground.shape[0] != 1:
			raise ValueError("generate_heatmap currently expects batch size 1 for both inputs.")

		satellite = satellite.to(self.device)
		ground = ground.to(self.device)

		# Step 1: encode reference with NO hook active yet.
		# LayerCAM.__init__ registers a permanent forward hook on target_layer.
		# If the hook existed here it would fire on this no_grad pass and crash
		# (output.register_hook requires requires_grad=True).
		if explain_view == "satellite":
			ref_emb = self._encode_single(ground)
			explained_input = satellite.detach().requires_grad_(True)
		else:
			ref_emb = self._encode_single(satellite)
			explained_input = ground.detach().requires_grad_(True)

		# Step 2: create a fresh wrapper + extractor AFTER reference encoding.
		# The hook is now registered only for the explained forward pass.
		wrapper = _CrossViewSimilarityWrapper(
			base_model=self.model,
			use_logit_scale=self.use_logit_scale,
		).to(self.device)
		cam_extractor = LayerCAM(wrapper, target_layer=self.target_layer)

		try:
			wrapper.zero_grad(set_to_none=True)
			with torch.enable_grad():
				scores = wrapper(explained_input, ref_emb)
			cam_list = cam_extractor(class_idx=0, scores=scores)
		finally:
			# Always remove hooks so they never bleed into subsequent calls.
			cam_extractor.remove_hooks()

		if not cam_list:
			raise RuntimeError("LayerCAM returned an empty CAM list.")

		cam_tensor = cam_list[0][0] if cam_list[0].dim() == 3 else cam_list[0]
		cam_map = cam_tensor.detach().cpu().numpy()

		source = satellite[0] if explain_view == "satellite" else ground[0]
		source_img = self._to_uint8_rgb(source)
		h, w = source_img.shape[:2]

		cam_map = cv2.resize(cam_map, (w, h), interpolation=cv2.INTER_CUBIC)
		cam_map = self._normalize_map(cam_map)

		heatmap = cv2.applyColorMap((cam_map * 255.0).astype(np.uint8), cv2.COLORMAP_JET)
		heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

		overlay = cv2.addWeighted(source_img, 1.0 - alpha, heatmap, alpha, 0.0)

		return HeatmapResult(
			image=source_img,
			heatmap=heatmap,
			overlay=overlay,
			score=float(scores[0, 0].detach().cpu().item()),
		)

	def generate_pair_heatmaps(
		self,
		satellite: torch.Tensor,
		ground: torch.Tensor,
		alpha: float = 0.45,
	) -> Dict[str, HeatmapResult]:
		return {
			"satellite": self.generate_heatmap(satellite, ground, explain_view="satellite", alpha=alpha),
			"ground": self.generate_heatmap(satellite, ground, explain_view="ground", alpha=alpha),
		}

	def close(self) -> None:
		# Hooks are removed per-call inside generate_heatmap; nothing to do here.
		pass


# ── Standalone test utility ────────────────────────────────────────────────────


def _load_and_preprocess(image_path: str, transforms) -> torch.Tensor:
	"""Read an RGB image from disk, apply albumentations transforms, return [1, C, H, W] tensor."""
	from PIL import Image as PILImage

	img = np.array(PILImage.open(image_path).convert("RGB"), dtype=np.uint8)
	result = transforms(image=img)
	return result["image"].unsqueeze(0)  # [1, C, H, W]


def run_heatmap_test(
	aerial_path: str,
	ground_path: str,
	weights_path: str,
	model_name: str = "convnext_base.fb_in22k_ft_in1k_384",
	img_size: int = 384,
	output_dir: str = "heatmap_outputs",
	alpha: float = 0.45,
	device: Optional[str] = None,
) -> None:
	"""
	Load pretrained Sample4Geo weights, run LayerCAM on one aerial/ground pair,
	and save the resulting heatmap overlays to output_dir.

	Saved files per view (satellite / ground):
	    <view>_original.png  – denormalized source image
	    <view>_heatmap.png   – raw JET colormap CAM
	    <view>_overlay.png   – CAM blended onto source image

	Args:
		aerial_path:  Path to the satellite/aerial image file.
		ground_path:  Path to the ground-level image file.
		weights_path: Path to a .pth checkpoint saved by Sample4Geo training.
		model_name:   timm model identifier (must match the checkpoint).
		img_size:     Resize both images to (img_size × img_size).
		output_dir:   Directory where output images are saved.
		alpha:        Heatmap blending factor (0 = original only, 1 = heatmap only).
		device:       'cuda', 'cpu', or None (auto-select).
	"""
	from pathlib import Path
	from PIL import Image as PILImage
	from sample4geo.model import TimmModel
	from sample4geo.transforms import get_transforms_val

	# ── Device ────────────────────────────────────────────────────────────────
	device_str = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
	dev = torch.device(device_str)
	print(f"Using device: {dev}")

	# ── Model ─────────────────────────────────────────────────────────────────
	print(f"Loading model '{model_name}' …")
	model = TimmModel(model_name, pretrained=False, img_size=img_size)

	print(f"Loading weights from '{weights_path}' …")
	state_dict = torch.load(weights_path, map_location="cpu")
	model.load_state_dict(state_dict, strict=False)
	model.to(dev).eval()

	# ── Transforms (identical to eval scripts) ────────────────────────────────
	data_config = model.get_config()
	mean = data_config["mean"]
	std = data_config["std"]
	sat_transforms, grd_transforms = get_transforms_val(
		image_size_sat=(img_size, img_size),
		img_size_ground=(round((224 / 1232) * img_size*2), img_size*2),
		mean=mean,
		std=std,
	)
    
	# ── Load & preprocess images ───────────────────────────────────────────────
	print(f"Preprocessing aerial  image: {aerial_path}")
	sat_tensor = _load_and_preprocess(aerial_path, sat_transforms)

	print(f"Preprocessing ground  image: {ground_path}")
	grd_tensor = _load_and_preprocess(ground_path, grd_transforms)
	# ── Run LayerCAM ───────────────────────────────────────────────────────────
	visualizer = CrossViewLayerCAMVisualizer(
		model=model,
		mean=mean,
		std=std,
		device=dev,
	)

	print("Generating heatmaps …")
	results = visualizer.generate_pair_heatmaps(sat_tensor, grd_tensor, alpha=alpha)
	visualizer.close()

	# ── Save outputs ───────────────────────────────────────────────────────────
	out_dir = Path(output_dir)
	out_dir.mkdir(parents=True, exist_ok=True)
	image_id = aerial_path.split("/")[-1].split(".")[0]
	for view, result in results.items():
		# PILImage.fromarray(result.image).save(out_dir / f"{image_id}_{view}_original.png")
		# PILImage.fromarray(result.heatmap).save(out_dir / f"{image_id}_{view}_heatmap.png")
		PILImage.fromarray(result.overlay).save(out_dir / f"{image_id}_{view}_overlay.png")
		print(
			f"  [{view}] similarity score: {result.score:.4f}"
			f"  →  {out_dir / f'{image_id}_{view}_{{original,heatmap,overlay}}.png'}"
		)

	print("Done.")


if __name__ == "__main__":
	import argparse

	parser = argparse.ArgumentParser(
		description="Generate LayerCAM heatmaps for a Sample4Geo aerial/ground image pair."
	)
	parser.add_argument("aerial", help="Path to the aerial (satellite) image.")
	parser.add_argument("ground", help="Path to the ground-level image.")
	parser.add_argument("weights", help="Path to the Sample4Geo .pth checkpoint.")
	parser.add_argument(
		"--model",
		default="convnext_base.fb_in22k_ft_in1k_384",
		help="timm model name that matches the checkpoint (default: convnext_base.fb_in22k_ft_in1k_384).",
	)
	parser.add_argument("--img-size", type=int, default=384, help="Input image size (default: 384).")
	parser.add_argument(
		"--output-dir",
		default="heatmap_outputs",
		help="Directory to save output images (default: heatmap_outputs).",
	)
	parser.add_argument(
		"--alpha",
		type=float,
		default=0.45,
		help="Heatmap overlay blending factor in [0, 1] (default: 0.45).",
	)
	parser.add_argument(
		"--device",
		default=None,
		help="Force 'cuda' or 'cpu'. Defaults to CUDA when available.",
	)

	args = parser.parse_args()
	run_heatmap_test(
		aerial_path=args.aerial,
		ground_path=args.ground,
		weights_path=args.weights,
		model_name=args.model,
		img_size=args.img_size,
		output_dir=args.output_dir,
		alpha=args.alpha,
		device=args.device,
	)


