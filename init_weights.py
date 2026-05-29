import torch.nn as nn
import math

from Swin_V2 import PatchPartition, SwinTransformerBlock, PatchMerging
from Faster_RCNN import RPN
from Fast_RCNN import FastRCNN



def initialize_weights_Swin(module):
    # PatchPartition initialization
    if isinstance(module, PatchPartition):
        nn.init.xavier_uniform_(module.conv2d.weight)
        if module.conv2d.bias is not None:
            nn.init.constant_(module.conv2d.bias, 0)

    # SwinTransformerBlock initialization
    elif isinstance(module, SwinTransformerBlock):
        nn.init.xavier_uniform_(module.query_proj.weight)
        if module.query_proj.bias is not None:
            nn.init.constant_(module.query_proj.bias, 0)
        nn.init.xavier_uniform_(module.key_proj.weight)
        if module.key_proj.bias is not None:
            nn.init.constant_(module.key_proj.bias, 0)
        nn.init.xavier_uniform_(module.value_proj.weight)
        if module.value_proj.bias is not None:
            nn.init.constant_(module.value_proj.bias, 0)
        nn.init.xavier_uniform_(module.output_proj.weight)
        if module.output_proj.bias is not None:
            nn.init.constant_(module.output_proj.bias, 0)
        nn.init.kaiming_normal_(module.fc1.weight, mode='fan_in', nonlinearity='relu')
        if module.fc1.bias is not None:
            nn.init.constant_(module.fc1.bias, 0)
        nn.init.xavier_uniform_(module.fc2.weight)
        if module.fc2.bias is not None:
            nn.init.constant_(module.fc2.bias, 0)

    # PatchMerging initialization
    elif isinstance(module, PatchMerging):
        nn.init.xavier_uniform_(module.reduce_dimensions.weight)
        if module.reduce_dimensions.bias is not None:
            nn.init.constant_(module.reduce_dimensions.bias, 0)
        
    # Norm layer initialization
    elif isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.LayerNorm):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)


def initialize_weights_FasterRCNN_rpn(module):
    pi = 0.33 # with downsampling, the positive/negative ratio is around 1:3
    # RPN initialization
    if isinstance(module, RPN):
        nn.init.kaiming_normal_(module.rpn_conv.weight, mode='fan_in', nonlinearity='relu')
        if module.rpn_conv.bias is not None:
            nn.init.constant_(module.rpn_conv.bias, 0)
        nn.init.normal_(module.rpn_objectness.weight, std=0.01)
        if module.rpn_objectness.bias is not None:
            nn.init.constant_(module.rpn_objectness.bias, -math.log((1 - pi) / pi))
        nn.init.normal_(module.rpn_bbox_pred.weight, std=0.01)
        if module.rpn_bbox_pred.bias is not None:
            nn.init.constant_(module.rpn_bbox_pred.bias, 0)


def initialize_weights_FasterRCNN_detector(module):
    # FastRCNN initialization
    if isinstance(module, FastRCNN):
        nn.init.kaiming_normal_(module.fc1.weight, mode='fan_in', nonlinearity='relu')
        if module.fc1.bias is not None:
            nn.init.constant_(module.fc1.bias, 0)
        nn.init.kaiming_normal_(module.fc2.weight, mode='fan_in', nonlinearity='relu')
        if module.fc2.bias is not None:
            nn.init.constant_(module.fc2.bias, 0)
        nn.init.normal_(module.cls_head.weight, std=0.01)
        if module.cls_head.bias is not None:
            nn.init.constant_(module.cls_head.bias, 0)
        nn.init.normal_(module.bbox_head.weight, std=0.01)
        if module.bbox_head.bias is not None:
            nn.init.constant_(module.bbox_head.bias, 0)
