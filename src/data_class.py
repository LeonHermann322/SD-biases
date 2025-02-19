from typing import Optional
import torch


class EmbeddingDataClass:
    def __init__(
        self,
        prefix: str,
        object: str,
        setting: str,
        embedding: torch.Tensor,
        cluster: Optional[int],
        reduced_dim_embedding: Optional[torch.Tensor],
    ):
        self.prefix = prefix
        self.object = object
        self.setting = setting
        self.embedding = embedding
        self.cluster = cluster
        self.reduced_dim_embedding = reduced_dim_embedding
