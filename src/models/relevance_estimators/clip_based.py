import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer, util
import numpy as np
from typing import Callable
from src.utils.image_processing import get_image

AVAILABLE_MODELS = (
    'clip-ViT-B-32',
    'clip-ViT-B-16',
    'clip-ViT-L-14'
)

class RelevanceEstimator(nn.Module):

    def __init__(self, config, dataset) -> None:
        super(RelevanceEstimator, self).__init__()
        self.config = config
        self.dataset = dataset
        self.model_name = self.config.dcg.relevance_estimator.name
        assert self.model_name in AVAILABLE_MODELS
        self.backbone = SentenceTransformer(self.model_name)
        self.sim_func = self.get_sim_score()

    def encode(self, x, **kwargs) -> np.ndarray:
        return self.backbone.encode(x, **kwargs)

    def get_sim_score(self) -> Callable:
        if self.config.dcg.relevance_estimator.sim_score == 'cosine':
            return util.cos_sim
        else:
            raise NotImplementedError

    def compute_relevance_estimation_t2i(self, query: str, document: str) -> float:
        assert type(query) == type(document)
        img = get_image(config=self.config, filename=document)
        query = query[:self.config.model.max_seq_length]
        query_emb = self.encode(query)
        document_emb = self.encode(img)
        return round(self.sim_func(query_emb, document_emb).item(), 4)

    def get_caption(self, caption_id: int) -> str:
        return self.dataset.captions[caption_id]['raw']

    def compute_relevance_estimation_i2t(self, query: str, document: str, caption_id: int) -> float:
        query_emb = self.encode(query)
        caption = self.get_caption(caption_id=caption_id)[:self.config.model.max_seq_length]
        caption_emb = self.encode(caption)
        return round(self.sim_func(query_emb, caption_emb).item(), 4)
