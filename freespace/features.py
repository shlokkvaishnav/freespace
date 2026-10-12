"""Image features. One small model (CLIP ViT-B/16) does all learned work:
image retrieval, crop matching and object names. Plus classic keypoint matching."""
from __future__ import annotations

from functools import lru_cache

import cv2
import numpy as np
import torch

CLIP_NAME = "openai/clip-vit-base-patch16"


def device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@lru_cache(maxsize=1)
def clip():
    from transformers import CLIPModel, CLIPProcessor

    return CLIPModel.from_pretrained(CLIP_NAME).eval().to(device()), CLIPProcessor.from_pretrained(CLIP_NAME)


@torch.no_grad()
def embed(images: list[np.ndarray], batch: int = 32) -> np.ndarray:
    """L2-normalised CLIP image embeddings for RGB uint8 images."""
    model, proc = clip()
    out = []
    for i in range(0, len(images), batch):
        x = proc(images=list(images[i:i + batch]), return_tensors="pt").to(device())
        f = model.get_image_features(**x).pooler_output
        out.append(torch.nn.functional.normalize(f, dim=-1).cpu().numpy())
    return np.concatenate(out)


@lru_cache(maxsize=4)
def _text_bank(words: tuple[str, ...]) -> torch.Tensor:
    model, proc = clip()
    feats = []
    with torch.no_grad():
        for i in range(0, len(words), 256):
            t = proc(text=[f"a photo of a {w}" for w in words[i:i + 256]], return_tensors="pt", padding=True).to(device())
            feats.append(torch.nn.functional.normalize(model.get_text_features(**t).pooler_output, dim=-1))
    return torch.cat(feats)


def name_scores(image_emb: np.ndarray, words: tuple[str, ...]) -> np.ndarray:
    """Softmax over the word list for one (averaged) image embedding."""
    t = _text_bank(words)
    e = torch.from_numpy(image_emb).float().to(t.device)
    e = torch.nn.functional.normalize(e, dim=-1)
    return (100 * t @ e).softmax(-1).cpu().numpy()


# ---------------------------------------------------------------- local matching

class SiftMatcher:
    name = "sift"

    def __init__(self, max_kp: int = 4000):
        self.sift = cv2.SIFT_create(nfeatures=max_kp)
        self.bf = cv2.BFMatcher(cv2.NORM_L2)

    def __call__(self, im0: np.ndarray, im1: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        k0, d0 = self.sift.detectAndCompute(cv2.cvtColor(im0, cv2.COLOR_RGB2GRAY), None)
        k1, d1 = self.sift.detectAndCompute(cv2.cvtColor(im1, cv2.COLOR_RGB2GRAY), None)
        if d0 is None or d1 is None or len(k0) < 8 or len(k1) < 8:
            return np.zeros((0, 2)), np.zeros((0, 2))
        pairs = self.bf.knnMatch(d0, d1, k=2)
        good = [m for m, n in (p for p in pairs if len(p) == 2) if m.distance < 0.8 * n.distance]
        p0 = np.array([k0[m.queryIdx].pt for m in good]).reshape(-1, 2)
        p1 = np.array([k1[m.trainIdx].pt for m in good]).reshape(-1, 2)
        return p0, p1


class LightGlueMatcher:
    name = "lightglue"

    def __init__(self, max_kp: int = 2048):
        from lightglue import LightGlue, SuperPoint

        self.dev = device()
        self.extractor = SuperPoint(max_num_keypoints=max_kp).eval().to(self.dev)
        self.matcher = LightGlue(features="superpoint").eval().to(self.dev)

    @torch.no_grad()
    def __call__(self, im0: np.ndarray, im1: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        from lightglue.utils import rbd

        t0 = torch.from_numpy(im0).permute(2, 0, 1).float().div(255).to(self.dev)
        t1 = torch.from_numpy(im1).permute(2, 0, 1).float().div(255).to(self.dev)
        f0, f1 = self.extractor.extract(t0), self.extractor.extract(t1)
        m = self.matcher({"image0": f0, "image1": f1})
        f0, f1, m = rbd(f0), rbd(f1), rbd(m)
        idx = m["matches"].cpu().numpy()
        return f0["keypoints"].cpu().numpy()[idx[:, 0]], f1["keypoints"].cpu().numpy()[idx[:, 1]]


def make_matcher(kind: str = "auto"):
    if kind in ("auto", "lightglue"):
        try:
            return LightGlueMatcher()
        except Exception:
            if kind == "lightglue":
                raise
    return SiftMatcher()
