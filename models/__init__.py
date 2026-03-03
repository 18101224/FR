from .vit_kprpe import ViTKPRPEModel
from omegaconf import OmegaConf
from .classifier import get_classifier

__all__ = ['get_model', 'get_classifier']

def get_model(args):
    if args.architecture in ['kprpe_base', 'kprpe_small']:
        model_size = "small" if args.architecture == "kprpe_small" else "base"
        config = OmegaConf.load(f"models/vit_kprpe/configs/v1_{model_size}_kprpe_splithead_unshared.yaml")
        config.output_dim = args.embedding_dim
        model = ViTKPRPEModel.from_config(config, runtime_args=args)
    else:
        raise ValueError(f'Invalid architecture: {args.architecture}')
    return model 
