import torch
from pathlib import Path
from typing import Optional, Tuple, List

from Swin_V2 import SwinTransformer
from init_weights import initialize_weights_Swin, initialize_weights_FasterRCNN


class SwinTransformerClassification(torch.nn.Module):
    def __init__(self, 
                backbone_weights_path: Path = None,
                backbone_input_size: Tuple[int, int] = (512, 512),
                backbone_patch_size: int = 4,
                backbone_patch_merging_ratio: int = 2,
                backbone_in_channels: int = 3,
                backbone_layers: list = [2, 2, 18, 2],
                backbone_query_size: int = 32, 
                backbone_n_heads: list = [3, 6, 12, 24], 
                backbone_mlp_factor: int = 4,
                backbone_window_size: int = 7, 
                n_classes: int = 2
                ):
        """
        An implementation of object detection model with Swin Transformer backbone and fasterRCNN head.
        params: 
            backbone_weights_path: Path to the backbone weights
            backbone_input_size: In Swin, the size of the input images. It is used to compute the number of patches and the hidden dims of the head.
            backbone_patch_size: In Swin, the number of pixels of a row and column to concat to create a patch. 4 => square of 16 pixels.
            backbone_patch_merging_ratio: In Swin, the number of patch to merge between each stage.
            backbone_in_channels: the number of channels of the input of the backbone
            backbone_layers: In Swin, the number of blocks for each stage
            backbone_query_size: In Swin, the size of the keys, queries and values. Multiply by the number of heads to get the hidden size
            backbone_n_heads: In Swin, the number of heads for each stage. Multiply by the query size to get the hidden dims
            backbone_mlp_factor: In Swin, the factor to multiply the hidden dims to get the MLP size in each block.
            backbone_window_size: In Swin, the size of the window (local attention)
            n_classes: number of classes to predict
        """
        super(SwinTransformerClassification, self).__init__()

        self.backbone = SwinTransformer(
            patch_size=backbone_patch_size,
            patch_merging_ratio=backbone_patch_merging_ratio,
            in_channels=backbone_in_channels,
            layers=backbone_layers,
            query_size=backbone_query_size,
            n_heads=backbone_n_heads,
            mlp_factor=backbone_mlp_factor,
            window_size=backbone_window_size
            )

        self.classif_head = torch.nn.Linear(backbone_n_heads[-1] * backbone_query_size, n_classes)

        # load pretrained backbone weights otherwise initialize
        if backbone_weights_path:
            self.backbone.load_state_dict(torch.load(backbone_weights_path))
        else:
            self.backbone.apply(initialize_weights_Swin)

        # initialize head weights
        torch.nn.init.xavier_uniform_(self.classif_head.weight)
        if self.classif_head.bias is not None:
            torch.nn.init.constant_(self.classif_head.bias, 0)

    def forward(self, input):
        x = self.backbone(input)
        x = x.mean(dim=[1, 2])
        x = self.classif_head(x)
        return x

if __name__ == "__main__":
    
    def count_param(module):
        total = 0
        for param in module.parameters():
            total += param.numel()
        return total

    model = SwinTransformerClassification(
        backbone_weights_path = None,
        backbone_input_size = (512, 512),
        backbone_patch_size = 4,
        backbone_patch_merging_ratio = 2,
        backbone_in_channels = 3,
        backbone_layers = [2, 2, 2, 2],
        backbone_query_size = 32,
        backbone_n_heads = [2, 4, 8, 16],
        backbone_mlp_factor=4,
        backbone_window_size = 7,
        n_classes = 100,
        )
    print(f"Number of parameters: {count_param(model)}")
