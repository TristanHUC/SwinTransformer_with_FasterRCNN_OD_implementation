import torch
from pathlib import Path
from typing import Optional, Tuple, List

from Swin_V2 import SwinTransformer
from init_weights import initialize_weights_Swin, initialize_weights_FasterRCNN
from Faster_RCNN import Faster_RCNN


class SwinTransformerDetection(torch.nn.Module):
    def __init__(self, 
                backbone_weights_path: Path = None,
                backbone_patch_size: int = 4,
                backbone_patch_merging_ratio: int = 2,
                backbone_in_channels: int = 3,
                backbone_layers: list = [2, 2, 18, 2],
                backbone_query_size: int = 32, 
                backbone_n_heads: list = [3, 6, 12, 24], 
                backbone_mlp_factor: int = 4,
                backbone_window_size: int = 7, 
                head_weights_path: Path = None,
                head_scales: List[int] = [64,128,256],
                head_shapes: List[List[int]] = [[1,1],[1,2],[2,1]],
                head_fixed_shape: Optional[Tuple[int, int, Tuple[int, int]]] = [16, 16, [512 / 16, 512 / 16]],
                head_objectness_iou_threshold_positive_boxes: float = 0.7,
                head_objectness_iou_threshold_negative_boxes: float = 0.3,
                head_xywh_format: bool = False,
                head_hidden_dim: int = 4096,
                n_classes: int = 2,
                background_label_id: int = 0
                ):
        """
        An implementation of object detection model with Swin Transformer backbone and fasterRCNN head.
        params: 
            backbone_weights_path: Path to the backbone weights
            backbone_patch_size: In Swin, the number of pixels of a row and column to concat to create a patch. 4 => square of 16 pixels.
            backbone_patch_merging_ratio: In Swin, the number of patch to merge between each stage.
            backbone_in_channels: the number of channels of the input of the backbone
            backbone_layers: In Swin, the number of blocks for each stage
            backbone_query_size: In Swin, the size of the keys, queries and values. Multiply by the number of heads to get the hidden size
            backbone_n_heads: In Swin, the number of heads for each stage. Multiply by the query size to get the hidden dims
            backbone_mlp_factor: In Swin, the factor to multiply the hidden dims to get the MLP size in each block.
            backbone_window_size: In Swin, the size of the window (local attention)
            head_weights_path: Path to the head weighs
            head_scales: In FasterRCNN, the scale value for each side of the anchors boxes to generate anchors.
            head_shapes: In FasterRCNN, the shapes of each anchors before multiplying by the scales.
            head_fixed_shape: In FasterRCNN, it's the size of the feature maps given to the head and the ratio to retrieve the RoI in the original image. If set we compute anchors once and apply them for each input. If not set, it must be given for each input.
            head_objectness_iou_threshold_positive_boxes: In FasterRCNN, threshold to consider a positive anchors in training and threshold to keep a proposal in inference.
            head_objectness_iou_threshold_negative_boxes: In FasterRCNN, threshold to consider a negative anchors in training. Useless in inference.
            head_xywh_format: If set, the dataset ground truth values will be considered of format x_min, y_min, width, height. If not set, the dataset will be considered of format i, j, height, width. It also determined the format of the predicted bounding boxes.
            head_hidden_dim: The hidden dimension for the detection head.
            n_classes: number of classes to predict.
            background_label_id: the class id for background 
        """
        super(SwinTransformerDetection, self).__init__()

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

        self.OD_head = Faster_RCNN(
            in_channels=backbone_patch_merging_ratio * len(backbone_layers) * backbone_n_heads[0] * backbone_query_size,
            hidden_dim=head_hidden_dim,
            n_class=n_classes,
            scales=head_scales,
            shapes=head_shapes,
            fixed_shape=head_fixed_shape,
            objectness_iou_threshold_positive_boxes=head_objectness_iou_threshold_positive_boxes,
            objectness_iou_threshold_negative_boxes=head_objectness_iou_threshold_negative_boxes,
            xywh_format=head_xywh_format,
            background_label_id=background_label_id
            )

        # load pretrained backbone weights otherwise initialize
        if backbone_weights_path:
            self.backbone.load_state_dict(torch.load(backbone_weights_path))
            print(f"Backbone weights loaded from {backbone_weights_path}")
        else:
            self.backbone.apply(initialize_weights_Swin)

        # load pretrained head weights otherwise initialize
        if head_weights_path:
            self.OD_head.load_state_dict(torch.load(head_weights_path))
            print(f"Head weights loaded from {head_weights_path}")
        else:
            self.OD_head.apply(initialize_weights_FasterRCNN)

    def forward_train(self, input, BoundingBoxes, labels, rpn_training_only:bool=False, λ_detector_reg = 10, λ_rpn_reg = 0.25):

        x = self.backbone(input)
        return self.OD_head.forward_train(x, BoundingBoxes, labels, rpn_training_only=rpn_training_only, λ_detector_reg = λ_detector_reg, λ_rpn_reg = λ_rpn_reg)

    def forward(self, input):

        x = self.backbone(input)
        return self.OD_head(x)
    
    
    @staticmethod
    def postprocess_remove_low_confidence_boxes(confidence_score_threshold, confidence_score, ROIs, final_boxes, indice_class_selected):
        """ Remove boxes with low confidence scores """
        return Faster_RCNN.postprocess_remove_low_confidence_boxes(confidence_score_threshold, confidence_score, ROIs, final_boxes, indice_class_selected)

    @staticmethod
    def postprocess_match_label(label, confidence_score, ROIs, final_boxes, indice_class_selected):
        """ Keep bounding boxes corresponding to desired label """
        return Faster_RCNN.postprocess_match_label(label, confidence_score, ROIs, final_boxes, indice_class_selected)

    @staticmethod
    def postprocess_exclude_label(label, confidence_score, ROIs, final_boxes, indice_class_selected):
        """ Exclude bounding boxes corresponding to desired label """
        return Faster_RCNN.postprocess_exclude_label(label, confidence_score, ROIs, final_boxes, indice_class_selected)


    @staticmethod
    def postprocess_gather_by_image(confidence_score, ROIs, final_boxes, indice_class_selected):
        """ Gather the predicted boxes and labels by image in the batch (instead of concatenated) """
        return Faster_RCNN.postprocess_gather_by_image(confidence_score, ROIs, final_boxes, indice_class_selected)


if __name__=="__main__":

    batch_size = 1
    channels = 3
    h, w = 512, 512
    window_size = 7
    torch.set_printoptions(threshold=float('inf'))

    image_original = torch.arange(batch_size * channels * h * w).reshape(batch_size, channels, h, w).float()

    model = SwinTransformerDetection()
    print(image_original.shape)
    for param in model.parameters():
        param.data.fill_(1.0)
    result = model(image_original)
    print(result[0].shape)