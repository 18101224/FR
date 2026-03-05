from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from facenet_pytorch import MTCNN
from PIL import Image
from torchvision import transforms

from ..base import BaseAligner
from ..retinaface_aligner import aligner_helper


class MTCNNAligner(BaseAligner):
    """
    Offline / utility aligner that matches the run_v1 landmark-based similarity alignment path,
    but uses MTCNN to predict 5-point landmarks.
    """

    def __init__(self, detector: MTCNN, config=None):
        super().__init__(config)
        self.detector = detector
        self.output_size = int(getattr(config, "output_size", 112))
        self.select_largest = bool(getattr(config, "select_largest", False))
        self.register_buffer("_device_indicator", torch.zeros(1), persistent=False)
        self._to_tensor = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    @classmethod
    def from_config(cls, config):
        detector = MTCNN(
            image_size=int(getattr(config, "output_size", 112)),
            margin=0,
            min_face_size=int(getattr(config, "min_face_size", 20)),
            thresholds=list(getattr(config, "thresholds", [0.6, 0.7, 0.8])),
            factor=float(getattr(config, "factor", 0.709)),
            post_process=False,
            keep_all=bool(getattr(config, "keep_all", True)),
            select_largest=bool(getattr(config, "select_largest", False)),
            device=torch.device(getattr(config, "device", "cpu")),
        )
        model = cls(detector=detector, config=config)
        model.eval()
        return model

    def make_train_transform(self):
        return self._to_tensor

    def make_test_transform(self):
        return self._to_tensor

    @torch.no_grad()
    def align_pil_batch(self, images: Sequence[Image.Image]) -> Tuple[List[Optional[Image.Image]], torch.Tensor]:
        if len(images) == 0:
            empty_scores = torch.empty((0, 1), device=self._device_indicator.device)
            return [], empty_scores

        pil_images = [image.convert("RGB") for image in images]
        try:
            boxes, probs, landmarks = self.detector.detect(pil_images, landmarks=True)
        except ValueError as exc:
            # facenet-pytorch MTCNN can raise here when no face is found in the batch.
            if "expected a non-empty list of Tensors" not in str(exc):
                raise
            batch_size = len(pil_images)
            boxes = [None] * batch_size
            probs = [None] * batch_size
            landmarks = [None] * batch_size

        aligned_images: List[Optional[Image.Image]] = []
        scores: List[float] = []

        for image, image_boxes, image_probs, image_landmarks in zip(pil_images, boxes, probs, landmarks):
            aligned_image, score = self._align_single_pil(
                image=image,
                boxes=image_boxes,
                probs=image_probs,
                landmarks=image_landmarks,
            )
            aligned_images.append(aligned_image)
            scores.append(score)

        score_tensor = torch.tensor(scores, device=self._device_indicator.device, dtype=torch.float32).unsqueeze(-1)
        return aligned_images, score_tensor

    @torch.no_grad()
    def forward(self, x):
        assert x.ndim == 4
        assert x.shape[1] == 3

        pil_images = [self._tensor_to_pil(sample) for sample in x]
        aligned_images, score = self.align_pil_batch(pil_images)

        aligned_tensors: List[torch.Tensor] = []
        ldmks: List[torch.Tensor] = []
        aligned_ldmks: List[torch.Tensor] = []
        thetas: List[torch.Tensor] = []
        bboxes: List[torch.Tensor] = []

        reference_ldmk = aligner_helper.reference_landmark()
        reference_ldmk_norm = self._reference_landmark_normalized(x.shape[-1], x.shape[-2])

        for aligned_image in aligned_images:
            if aligned_image is None:
                resized = F.interpolate(
                    x.new_zeros((1, 3, self.output_size, self.output_size)),
                    size=(self.output_size, self.output_size),
                    mode="bilinear",
                    align_corners=True,
                ).squeeze(0)
                aligned_tensors.append(resized)
                ldmks.append(reference_ldmk_norm.clone())
                aligned_ldmks.append(torch.from_numpy(reference_ldmk).to(x.device).float() / float(self.output_size))
                thetas.append(self._identity_theta(x.device))
                bboxes.append(torch.tensor([0.0, 0.0, 1.0, 1.0], device=x.device))
                continue

            aligned_tensors.append(self._to_tensor(aligned_image).to(x.device))
            normalized_ldmk = torch.from_numpy(reference_ldmk).to(x.device).float()
            normalized_ldmk[:, 0] /= float(self.output_size)
            normalized_ldmk[:, 1] /= float(self.output_size)
            ldmks.append(normalized_ldmk)
            aligned_ldmks.append(normalized_ldmk)
            thetas.append(self._identity_theta(x.device))
            bboxes.append(torch.tensor([0.0, 0.0, 1.0, 1.0], device=x.device))

        return (
            torch.stack(aligned_tensors, dim=0),
            torch.stack(ldmks, dim=0),
            torch.stack(aligned_ldmks, dim=0),
            score.to(x.device),
            torch.stack(thetas, dim=0),
            torch.stack(bboxes, dim=0),
        )

    def _align_single_pil(
        self,
        image: Image.Image,
        boxes: Optional[np.ndarray],
        probs: Optional[np.ndarray],
        landmarks: Optional[np.ndarray],
    ) -> Tuple[Optional[Image.Image], float]:
        selected = self._select_face(boxes=boxes, probs=probs, landmarks=landmarks)
        if selected is None:
            return None, 0.0

        _, _, selected_landmarks = selected
        image_tensor = self._to_tensor(image).unsqueeze(0)
        ldmk_tensor = torch.from_numpy(selected_landmarks.reshape(1, 10)).float()
        ldmk_tensor[:, 0::2] /= float(image.width)
        ldmk_tensor[:, 1::2] /= float(image.height)

        reference_ldmk = aligner_helper.reference_landmark()
        cv2_tfms = aligner_helper.get_cv2_affine_from_landmark(
            ldmk_tensor,
            reference_ldmk,
            image.width,
            image.height,
        )
        theta = aligner_helper.cv2_param_to_torch_theta(
            cv2_tfms,
            image.width,
            image.height,
            self.output_size,
            self.output_size,
        )
        grid = F.affine_grid(
            theta,
            torch.Size((1, 3, self.output_size, self.output_size)),
            align_corners=True,
        )
        aligned = F.grid_sample(image_tensor + 1.0, grid, align_corners=True) - 1.0
        return self._tensor_to_pil(aligned[0]), float(selected[1])

    def _select_face(
        self,
        boxes: Optional[np.ndarray],
        probs: Optional[np.ndarray],
        landmarks: Optional[np.ndarray],
    ) -> Optional[Tuple[np.ndarray, float, np.ndarray]]:
        if boxes is None or landmarks is None:
            return None
        if len(boxes) == 0 or len(landmarks) == 0:
            return None

        if probs is None:
            probs = np.ones((len(boxes),), dtype=np.float32)

        if self.select_largest:
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            best_index = int(np.argmax(areas))
        else:
            best_index = int(np.argmax(probs))

        return boxes[best_index], float(probs[best_index]), landmarks[best_index].astype(np.float32)

    def _tensor_to_pil(self, tensor: torch.Tensor) -> Image.Image:
        image = tensor.detach().cpu().float().clamp(-1.0, 1.0)
        image = image * 0.5 + 0.5
        image = image.clamp(0.0, 1.0)
        image = transforms.ToPILImage()(image)
        return image.convert("RGB")

    def _identity_theta(self, device: torch.device) -> torch.Tensor:
        return torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], device=device, dtype=torch.float32)

    def _reference_landmark_normalized(self, width: int, height: int) -> torch.Tensor:
        reference = torch.from_numpy(aligner_helper.reference_landmark()).float()
        reference[:, 0] /= float(width)
        reference[:, 1] /= float(height)
        return reference
