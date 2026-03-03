from __future__ import annotations

from .classifiers.base import BaseClassifier
from .classifiers.partial_fc.partial_fc import PartialFC_V2


def get_classifier(sample_rate, margin_loss_fn, output_dim, num_classes, rank, world_size):
    return PartialFCClassifier.build(
        sample_rate=sample_rate,
        margin_loss_fn=margin_loss_fn,
        output_dim=output_dim,
        num_classes=num_classes,
        rank=rank,
        world_size=world_size,
    )


class PartialFCClassifier(BaseClassifier):
    def __init__(self, classifier, rank, world_size):
        super(PartialFCClassifier, self).__init__()
        self.partial_fc = classifier
        self.rank = rank
        self.world_size = world_size
        self.apply_ddp = False

    @classmethod
    def build(cls, sample_rate, margin_loss_fn, output_dim, num_classes, rank, world_size):
        classifier = PartialFC_V2(
            rank=rank,
            world_size=world_size,
            margin_loss=margin_loss_fn,
            embedding_size=output_dim,
            num_classes=num_classes,
            sample_rate=sample_rate,
        )
        model = cls(classifier, rank, world_size)
        model.eval()
        return model

    def forward(self, local_embeddings, local_labels):
        return self.partial_fc(local_embeddings, local_labels)
