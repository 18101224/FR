from typing import Callable
import torch
from torch import distributed
import torch.nn.functional as F
from torch.nn.functional import linear, normalize
from losses.adaface import AdaFaceLoss



class PartialFC_V2(torch.nn.Module):
    """
    https://arxiv.org/abs/2203.15565
    A distributed sparsely updating variant of the FC layer, named Partial FC (PFC).
    When sample rate less than 1, in each iteration, positive class centers and a random subset of
    negative class centers are selected to compute the margin-based softmax loss, all class
    centers are still maintained throughout the whole training process, but only a subset is
    selected and updated in each iteration.
    .. note::
        When sample rate equal to 1, Partial FC is equal to model parallelism(default sample rate is 1).
    Example:
    --------
    >>> module_pfc = PartialFC(embedding_size=512, num_classes=8000000, sample_rate=0.2)
    >>> for img, labels in data_loader:
    >>>     embeddings = net(img)
    >>>     loss = module_pfc(embeddings, labels)
    >>>     loss.backward()
    >>>     optimizer.step()
    """
    _version = 2

    def __init__(
        self,
        rank: int,
        world_size: int,
        margin_loss: Callable,
        embedding_size: int,
        num_classes: int,
        sample_rate: float = 1.0,
    ):
        """
        Paramenters:
        -----------
        embedding_size: int
            The dimension of embedding, required
        num_classes: int
            Total number of classes, required
        sample_rate: float
            The rate of negative centers participating in the calculation, default is 1.0.
        """
        super(PartialFC_V2, self).__init__()
        self.rank = rank
        self.world_size = world_size
        self.dist_cross_entropy = DistCrossEntropy()
        self.embedding_size = embedding_size
        self.sample_rate: float = sample_rate

        # make num_class divisible by self.world_size for ddp
        _num_classes = num_classes // self.world_size * self.world_size
        if _num_classes < num_classes:
            _num_classes = _num_classes + self.world_size
        num_classes = _num_classes
        self.num_classes: int = num_classes

        self.num_local: int = num_classes // self.world_size + int(
            self.rank < num_classes % self.world_size
        )

        # for i in range(8):
        #     num_local = (num_classes // self.world_size + int( i < num_classes % self.world_size ))
        #     class_start = num_classes // self.world_size * i + min( i, num_classes % self.world_size )
        #     print(num_local, class_start)

        self.class_start: int = num_classes // self.world_size * self.rank + min(
            self.rank, num_classes % self.world_size
        )
        self.num_sample: int = int(self.sample_rate * self.num_local)
        self.last_batch_size: int = 0

        self.is_updated: bool = True
        self.init_weight_update: bool = True
        self.weight = torch.nn.Parameter(torch.normal(0, 0.01, (self.num_local, embedding_size)))

        # margin_loss
        if callable(margin_loss):
            self.margin_softmax = margin_loss
        else:
            raise

    def sample(self, labels, index_positive):
        """
            This functions will change the value of labels
            Parameters:
            -----------
            labels: torch.Tensor
                pass
            index_positive: torch.Tensor
                pass
            optimizer: torch.optim.Optimizer
                pass
        """
        with torch.no_grad():
            device = self.weight.device
            positive = torch.unique(labels[index_positive], sorted=True).to(device)
            if self.num_sample - positive.size(0) >= 0:
                perm = torch.rand(size=[self.num_local], device=device)
                perm[positive] = 2.0
                index = torch.topk(perm, k=self.num_sample)[1]
                index = index.sort()[0]
            else:
                index = positive
            self.weight_index = index

            labels[index_positive] = torch.searchsorted(index, labels[index_positive])

        return self.weight[self.weight_index]

    def forward(
        self,
        local_embeddings: torch.Tensor,
        local_labels: torch.Tensor,
    ):
        """
        Parameters:
        ----------
        local_embeddings: torch.Tensor
            feature embeddings on each GPU(Rank).
        local_labels: torch.Tensor
            labels on each GPU(Rank).
        Returns:
        -------
        loss: torch.Tensor
            pass
        """

        local_labels.squeeze_()
        local_labels = local_labels.long()

        batch_size = local_embeddings.size(0)
        if self.last_batch_size == 0:
            self.last_batch_size = batch_size
        assert self.last_batch_size == batch_size, (
            f"last batch size do not equal current batch size: {self.last_batch_size} vs {batch_size}")

        is_distributed = (
            self.world_size > 1
            and distributed.is_available()
            and distributed.is_initialized()
        )

        if is_distributed:
            _gather_embeddings = [
                torch.zeros((batch_size, self.embedding_size), dtype=local_embeddings.dtype, device=local_embeddings.device)
                for _ in range(self.world_size)
            ]
            _gather_labels = [
                torch.zeros(batch_size, device=local_labels.device, dtype=torch.long)
                for _ in range(self.world_size)
            ]
            _list_embeddings = AllGather(local_embeddings, *_gather_embeddings)
            distributed.all_gather(_gather_labels, local_labels)

            embeddings = torch.cat(_list_embeddings)
            labels = torch.cat(_gather_labels)
        else:
            embeddings = local_embeddings
            labels = local_labels

        labels = labels.view(-1, 1)
        index_positive = (self.class_start <= labels) & (
            labels < self.class_start + self.num_local
        )
        labels[~index_positive] = -1
        labels[index_positive] -= self.class_start

        if self.sample_rate < 1:
            weight = self.sample(labels, index_positive)
        else:
            weight = self.weight

        # with torch.cuda.amp.autocast(self.fp16):
        norms = embeddings.norm(p=2, dim=1, keepdim=True).clamp_min(1e-8)
        norm_embeddings = embeddings / norms

        norm_weight_activated = normalize(weight)
        logits = linear(norm_embeddings, norm_weight_activated)

        logits = logits.clamp(-1, 1)

        if isinstance(self.margin_softmax, AdaFaceLoss):
            logits = self.margin_softmax(logits=logits, labels=labels, norms=norms)
        else:
            raise ValueError('parital FC margin_softmax not supported type')

        if is_distributed:
            loss = self.dist_cross_entropy(logits, labels)
        else:
            loss = F.cross_entropy(logits, labels.view(-1))
        return loss


class DistCrossEntropyFunc(torch.autograd.Function):
    """
    CrossEntropy loss is calculated in parallel, allreduce denominator into single gpu and calculate softmax.
    Implemented of ArcFace (https://arxiv.org/pdf/1801.07698v1.pdf):
    """

    @staticmethod
    def forward(ctx, logits: torch.Tensor, label: torch.Tensor):
        """ """
        batch_size = logits.size(0)
        ctx.logits_dtype = logits.dtype
        logits_fp32 = logits.float()
        # for numerical stability
        max_logits, _ = torch.max(logits_fp32, dim=1, keepdim=True)
        # local to global
        distributed.all_reduce(max_logits, distributed.ReduceOp.MAX)
        logits_fp32.sub_(max_logits)
        logits_fp32.exp_()
        sum_logits_exp = torch.sum(logits_fp32, dim=1, keepdim=True)
        # local to global
        distributed.all_reduce(sum_logits_exp, distributed.ReduceOp.SUM)
        logits_fp32.div_(sum_logits_exp)
        index = torch.where(label != -1)[0]
        # loss
        loss = torch.zeros(batch_size, 1, device=logits.device, dtype=logits_fp32.dtype)
        loss[index] = logits_fp32[index].gather(1, label[index])
        distributed.all_reduce(loss, distributed.ReduceOp.SUM)
        ctx.save_for_backward(index, logits_fp32, label)
        return loss.clamp_min_(1e-30).log_().mean() * (-1)

    @staticmethod
    def backward(ctx, loss_gradient):
        """
        Args:
            loss_grad (torch.Tensor): gradient backward by last layer
        Returns:
            gradients for each input in forward function
            `None` gradients for one-hot label
        """
        (
            index,
            logits,
            label,
        ) = ctx.saved_tensors
        batch_size = logits.size(0)
        grad_logits = logits.clone()
        one_hot = torch.zeros(
            size=[index.size(0), logits.size(1)], device=logits.device, dtype=logits.dtype
        )
        one_hot.scatter_(1, label[index], 1)
        grad_logits[index] -= one_hot
        grad_logits.div_(batch_size)
        grad_logits.mul_(loss_gradient.to(dtype=grad_logits.dtype))
        if grad_logits.dtype != ctx.logits_dtype:
            grad_logits = grad_logits.to(dtype=ctx.logits_dtype)
        return grad_logits, None


class DistCrossEntropy(torch.nn.Module):
    def __init__(self):
        super(DistCrossEntropy, self).__init__()

    def forward(self, logit_part, label_part):
        return DistCrossEntropyFunc.apply(logit_part, label_part)


class AllGatherFunc(torch.autograd.Function):
    """AllGather op with gradient backward"""

    @staticmethod
    def forward(ctx, tensor, *gather_list):
        gather_list = list(gather_list)
        distributed.all_gather(gather_list, tensor)
        return tuple(gather_list)

    @staticmethod
    def backward(ctx, *grads):
        grad_list = list(grads)
        rank = distributed.get_rank()
        grad_out = grad_list[rank]

        dist_ops = [
            distributed.reduce(grad_out, rank, distributed.ReduceOp.SUM, async_op=True)
            if i == rank
            else distributed.reduce(
                grad_list[i], i, distributed.ReduceOp.SUM, async_op=True
            )
            for i in range(distributed.get_world_size())
        ]
        for _op in dist_ops:
            _op.wait()

        grad_out *= len(grad_list)  # cooperate with distributed loss function
        return (grad_out, *[None for _ in range(len(grad_list))])


AllGather = AllGatherFunc.apply
