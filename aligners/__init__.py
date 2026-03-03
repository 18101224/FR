import os
from types import SimpleNamespace

from omegaconf import OmegaConf

from .base import BaseAligner

def get_aligner(cfg_path):
    aligner_cfg = OmegaConf.load(os.path.join(cfg_path, 'aligner.yaml'))
    start_from = os.path.join(cfg_path, 'aligner.pt')
    aligner_cfg.start_from = start_from if os.path.exists(start_from) else ''
    if aligner_cfg.name == 'none':
        from .none import NoneAligner
        aligner = NoneAligner.from_config(aligner_cfg)
    elif aligner_cfg.name == 'retinaface_aligner':
        from .retinaface_aligner import RetinaFaceAligner
        aligner = RetinaFaceAligner.from_config(aligner_cfg)
    elif aligner_cfg.name == 'differentiable_face_aligner':
        from .differentiable_face_aligner import DifferentiableFaceAligner
        aligner = DifferentiableFaceAligner.from_config(aligner_cfg)
    elif aligner_cfg.name == 'mtcnn_aligner':
        from .mtcnn_aligner import MTCNNAligner
        aligner = MTCNNAligner.from_config(aligner_cfg)
    else:
        raise ValueError(f"Unknown classifier: {aligner_cfg.name}")

    if aligner_cfg.start_from:
        aligner.load_state_dict_from_path(aligner_cfg.start_from)

    for param in aligner.parameters():
        param.requires_grad = False
    aligner.eval()
    return aligner


def build_mtcnn_aligner(device: str = "cuda:0", output_size: int = 112) -> BaseAligner:
    from .mtcnn_aligner import MTCNNAligner

    aligner_cfg = SimpleNamespace(
        name="mtcnn_aligner",
        output_size=int(output_size),
        min_face_size=20,
        thresholds=[0.6, 0.7, 0.8],
        factor=0.709,
        select_largest=False,
        keep_all=True,
        freeze=True,
        start_from="",
        device=str(device),
    )
    aligner = MTCNNAligner.from_config(aligner_cfg).to(device)
    aligner.eval()
    return aligner
