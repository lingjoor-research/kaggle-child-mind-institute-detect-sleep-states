from typing import List

import torch
import torch.nn as nn


class MultiExtractor(nn.Module):
    def __init__(
        self,
        feature_extractors: List[nn.Module],
    ):
        super().__init__()
        self.feature_extractors = nn.ModuleList(feature_extractors)
        self.out_chans = sum([fe.out_chans for fe in feature_extractors])
        self.height = feature_extractors[0].height
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): (batch_size, in_channels, time_steps)

        Returns:
            torch.Tensor : (batch_size, out_chans, height, time_steps)
        """
        
        features = []
        for feature_extractor in self.feature_extractors:
            features.append(feature_extractor(x))
        features = torch.cat(features, dim=1)
        return features
    