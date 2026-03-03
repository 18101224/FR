import argparse
from torch.distributed import init_process_group
import os 
import torch 

__all__ = ['get_arguments']


def str2bool(value):
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")

def get_arguments() -> argparse.Namespace:
    args = argparse.ArgumentParser()

    #gpu settings
    args.add_argument('--num_workers', type=int, default=8)
    args.add_argument('--use_accelerator', type=str2bool, default=False)
    args.add_argument('--mixed_precision', type=str, default='no')
    args.add_argument('--use_flash_attn', type=str2bool, default=False)


    #training hyperparameters
    args.add_argument('--batch_size', type=int, default=128)
    args.add_argument('--n_epochs', type=int, default=30)
    args.add_argument('--learning_rate', type=float, default=1e-4)
    args.add_argument('--weight_decay', type=float, default=0.05)

    #checkpoint 
    args.add_argument('--ckpt_path', type=str, default=None,)
    args.add_argument('--resume_path', type=str, default=None)
    args.add_argument('--aligner_ckpt', type=str, default=None)

    #dataset hyperparameters
    args.add_argument('--dataset_name', type=str, default='ms1mv3')
    args.add_argument('--dataset_root', type=str, default='data/ms1mv3')



    #FR loss hyperparameters
    args.add_argument('--m', type=float, default=0.4, help='AdaFace Loss Margin')
    args.add_argument('--h', type=float, default=0.333, help='AdaFace Loss h')

    #model_hyperparameters
    args.add_argument('--architecture', type=str, default='kprpe_base', choices=['kprpe_base', 'kprpe_small'])
    args.add_argument('--embedding_dim', type=int, default=512) # 512 for kprpe_base and kprpe_small
    args.add_argument('--cf_sample_rate', type=float, default=1.0)
    
    args = args.parse_args()
    args.world_size = max(int(os.environ.get('WORLD_SIZE', '1')), 1)
    args.rank = 0
    args.local_rank = 0

    # Keep optimizer behavior fixed unless the code is edited.
    args.optimizer = 'AdamW'
    args.scheduler = 'cosine'
    args.warmup_epochs = 3
    args.steps_per_epoch = None

    if args.world_size > 1 and not args.use_accelerator:
        args.rank = int(os.environ['RANK'])
        args.local_rank = int(os.environ['LOCAL_RANK'])
        init_process_group(backend='nccl', init_method='env://', world_size=args.world_size, rank=args.rank)
        torch.cuda.set_device(args.local_rank)
        args.batch_size = int(args.batch_size // args.world_size)

    return args
