from typing import Optional, Tuple, List

import torch
from torch import nn
import torchvision.ops as ops
import math
from utils import loss_RPN, clip_bb_and_transform
from Fast_RCNN import FastRCNN

class RPN(nn.Module):
    def __init__(self, in_channels: int, bbox_normalize_stds: torch.Tensor, scales: List[int], shapes: List[List[int]], fixed_shape: Optional[Tuple[int, int, Tuple[int, int]]]=None):
        super(RPN, self).__init__()

        self.register_buffer("bbox_normalize_stds", bbox_normalize_stds)

        self.scales =  scales
        self.shapes = shapes
        self.num_anchors = len(shapes) * len(scales)

        self.rpn_conv = nn.Conv2d(in_channels, 256, kernel_size=(3,3), stride=1, padding='same')
        #self.rpn_bn = nn.GroupNorm(num_groups=32, num_channels=256) #nn.BatchNorm2d(256)
        self.rpn_activation = nn.ReLU()

        self.rpn_objectness = nn.Conv2d(256, self.num_anchors, kernel_size=(1,1), stride=1, padding='same')

        self.rpn_bbox_pred = nn.Conv2d(256, self.num_anchors * 4, kernel_size=(1,1), stride=1, padding='same')


        # if the image shape will always be the same => compute the base anchors only once
        if fixed_shape :
            self.fixed_shape = True
            anchors_boxes, valid_mask = self.create_anchors_boxes(fixed_shape[0], fixed_shape[1], fixed_shape[2])
            self.register_buffer("anchors_boxes", anchors_boxes)
            self.register_buffer("valid_mask", valid_mask)

    
    def create_anchors_boxes(self, H: int, W: int, upscale_factor: Tuple[int, int]):
        """
        create the anchor boxes from scratch. 
        """

        with torch.no_grad():

            # retrieve the upscale factor which allow to scale the bounding to the original image size
            upscale_factor_H, upscale_factor_W = upscale_factor

            # create a grid of indices as a base for of anchors
            position_H = torch.arange(0, H, 1)
            position_W = torch.arange(0, W, 1)

            grid = torch.stack(torch.meshgrid(position_H, position_W, indexing='ij'), dim=-1)

            # for each position: create N anchors
            anchors = []
            for scale in self.scales:
                for shape in self.shapes:
                    shape_tensor = torch.tensor(shape)
                    length = torch.tensor(scale * shape_tensor)

                    #shape : H, W, 4 : y_min, x_min, h, w
                    anchors.append(torch.stack([
                        (grid[:, :, 0] * upscale_factor_H + upscale_factor_H // 2) - (length[0] // 2),
                        (grid[:, :, 1] * upscale_factor_W + upscale_factor_W // 2) - (length[1] // 2),
                        torch.full((H, W), length[0]),
                        torch.full((H, W), length[1])
                    ], dim=-1))

            anchors = torch.stack(anchors, dim = 2) #shape : H, W, 9, 4

            # create a mask on the valid ones (ones within the image boundaries)
            height = H*upscale_factor_H
            width = W*upscale_factor_W

            y_min, x_min, h, w = anchors[..., 0], anchors[..., 1], anchors[..., 2], anchors[..., 3]
            y_max = y_min + h
            x_max = x_min + w

            valid_mask = (x_min >= 0) & (y_min >= 0) & (y_max <= height) & (x_max <= width)

            return anchors.view(-1, 4), valid_mask.view(-1)


    def forward(self, input, upscale_factor=None, train = False):
        B, _,  H, W = input.shape

        # If all inputs are of the same size : only compute reference anchors one time
        if self.fixed_shape:
            anchor_boxes = self.anchors_boxes
            if train == True :
                valid_mask = self.valid_mask
                anchor_boxes = anchor_boxes[valid_mask]
            else :
                valid_mask = torch.full([H * W * self.num_anchors], True, dtype=torch.bool, device=input.device)
        elif upscale_factor :
            if train == True :
                anchor_boxes, valid_mask = self.create_anchors_boxes(H, W, upscale_factor)
                anchor_boxes = anchor_boxes[valid_mask]
            else :
                anchor_boxes, _ = self.create_anchors_boxes(H, W, upscale_factor)
                valid_mask = torch.full([H * W * self.num_anchors], True, dtype=torch.bool)
            anchor_boxes = anchor_boxes.to(input.device)
            valid_mask = valid_mask.to(input.device)
        else :
            raise NotImplementedError(" please provide the upscale factor (ratio image_shape/feature_map_shape)")

        # one convolution layer
        x = self.rpn_conv(input)
        #x = self.rpn_bn(x)
        x = self.rpn_activation(x)

        # compute the objectness for each anchors and keep only the valid ones (valid == within the image range)
        objectness_list = self.rpn_objectness(x) # output shape: B, num_anchors, H, W
        objectness_score_map = objectness_list.permute(0, 2, 3, 1).contiguous().view(B, -1)[:, valid_mask]

        # compute the bounding boxes shifting values for each anchors and keep only the valid ones (valid == within the image range)
        bbox_list = self.rpn_bbox_pred(x) # output shape: B, num_anchors*4, H, W
        box_deltas_map = bbox_list.permute(0, 2, 3, 1).contiguous().view(B, -1, 4)[:, valid_mask]

        # un-normalize the deltas, normalized during training
        box_deltas_map_unscaled = box_deltas_map * self.bbox_normalize_stds

        anchor_y_min = anchor_boxes[..., 0]
        anchor_x_min = anchor_boxes[..., 1]
        anchor_h = anchor_boxes[..., 2]
        anchor_w = anchor_boxes[..., 3]

        # Convert anchors to (center_y, center_x, h, w)
        anchor_center_y = anchor_y_min + 0.5 * anchor_h
        anchor_center_x = anchor_x_min + 0.5 * anchor_w

        # The deltas predicted by the network
        dy = box_deltas_map_unscaled[..., 0]
        dx = box_deltas_map_unscaled[..., 1]
        dh = box_deltas_map_unscaled[..., 2]
        dw = box_deltas_map_unscaled[..., 3]

        # Clamping dh and dw to prevent overflow
        dh = torch.clamp(dh, max=math.log(1000. / 16.))
        dw = torch.clamp(dw, max=math.log(1000. / 16.)) 

        # Apply the deltas to the anchor boxes
        pred_center_y = dy * anchor_h + anchor_center_y
        pred_center_x = dx * anchor_w + anchor_center_x
        pred_h = anchor_h * torch.exp(dh)
        pred_w = anchor_w * torch.exp(dw)

        # Convert the refined boxes back to (x_min, y_min, w, h)
        y_new = pred_center_y - 0.5 * pred_h
        x_new = pred_center_x - 0.5 * pred_w
        proposals = torch.stack([y_new, x_new, pred_h, pred_w], dim=-1)

        return objectness_score_map, proposals, box_deltas_map, anchor_boxes


class Faster_RCNN(nn.Module):
    def __init__(self, in_channels: int, n_class: int, scales: List[int], shapes: List[List[int]],  fixed_shape: Optional[Tuple[int, int, Tuple[int, int]]] = None, objectness_iou_threshold_positive_boxes: float = 0.7, objectness_iou_threshold_negative_boxes: float = 0.3, xywh_format: bool = False, background_label_id: int = 0):
        super(Faster_RCNN, self).__init__()

        self.background_label_id = background_label_id
        self.objectness_iou_threshold_positive_boxes = objectness_iou_threshold_positive_boxes
        self.objectness_iou_threshold_negative_boxes = objectness_iou_threshold_negative_boxes
        self.xywh_format = xywh_format

        if fixed_shape is None:
            self.fixed_shape = False
            self.roi_pool = None
            self.upscale_factor = None
        else:
            self.fixed_shape = True
            self.upscale_factor = fixed_shape[2]
            self.roi_pool = ops.RoIAlign(output_size=(7, 7), spatial_scale=1 / self.upscale_factor[0], sampling_ratio=2, aligned=True)

        self.n_class = n_class

        bbox_normalize_stds = torch.tensor([0.1, 0.1, 0.2, 0.2])
        self.register_buffer("bbox_normalize_stds", bbox_normalize_stds)
        self.rpn = RPN(
            in_channels=in_channels,
            bbox_normalize_stds=bbox_normalize_stds,
            scales=scales,
            shapes=shapes,
            fixed_shape=fixed_shape)
        self.rpn_objectness_sigmoid = nn.Sigmoid()

        self.detector = FastRCNN(in_channels, n_class, hidden_dim=1024, roi_output_size=(7, 7))
        self.detector_softmax = nn.Softmax(dim=-1)


    def ensure_parameters(self, upscale_factor):
        if upscale_factor:
            upscale_factor = upscale_factor
            roi_pool = ops.RoIAlign(output_size=(7, 7), spatial_scale=1 / upscale_factor[0], sampling_ratio=2, aligned=True)
        elif self.fixed_shape:
            upscale_factor = self.upscale_factor
            roi_pool = self.roi_pool
        if not upscale_factor:
            raise ValueError("Please provide the upscale factor (ratio image_shape/feature_map_shape) either in the constructor if the image shape is fixed or in the forward_train method if the image shape can vary")
        return roi_pool, upscale_factor
        

    def roi_and_forward_detector_training(self, roi_pool, upscale_factor, batch_size, H, W, feature_map, proposals):
        # Clip the bounding box and transform h and w into x_max and y_max for torchvision ROI operation
        ROIs = []
        for image in range(batch_size):
            ROIs.append(clip_bb_and_transform(proposals[image], H * upscale_factor[0], W * upscale_factor[1]).float())

        # ROI pooling : extract feature map block
        pooled_features = roi_pool(feature_map.float(), ROIs)

        # Detector head part : classify the bounding boxes and refine them
        class_logits, bbox_deltas_map = self.detector.forward(pooled_features)
        return class_logits, bbox_deltas_map


    def forward_train(self, feature_map, GT_BB, labels, upscale_factor=None):

        roi_pool, upscale_factor = self.ensure_parameters(upscale_factor)

        B, _, H, W = feature_map.shape
        feature_map = feature_map.permute(0, 3, 1, 2).contiguous() #shape : B, C, H, W for torch convolution layers and torchvision ops

        # If the Ground Truth dataset is in (x_min, y_min, w, h) format, convert from xywh to yxhw for network compatibility
        if self.xywh_format:
            GT_x = GT_BB[..., 0]
            GT_y = GT_BB[..., 1]
            GT_w = GT_BB[..., 2]
            GT_h = GT_BB[..., 3]
            GT_BB = torch.stack([GT_y, GT_x, GT_h, GT_w], dim=-1)

        # RPN part : create relevant proposals from the feature maps
        preds = self.rpn.forward(feature_map, upscale_factor, train=True)
        loss_rpn, positive_proposals, negative_proposals, aligned_positive_proposals_GT_BB, aligned_positive_proposals_GT_labels = loss_RPN(preds, GT_BB, labels, self.bbox_normalize_stds, objectness_iou_threshold_positive=self.objectness_iou_threshold_positive_boxes, objectness_iou_threshold_negative=self.objectness_iou_threshold_negative_boxes)
        #print("overall RPN loss:", loss_rpn.item())

        # Proposal part:
        mixed_proposals = []
        aligned_GT_labels_list = []
        is_positive_list = []

        for i in range(B):
            num_neg = negative_proposals[i].shape[0]
            num_pos = positive_proposals[i].shape[0]
            # Combine proposals
            mixed_proposals.append(torch.cat([positive_proposals[i], negative_proposals[i]], dim=0))
            
            # Combine labels
            num_neg = negative_proposals[i].shape[0]
            aligned_GT_labels_list.append(aligned_positive_proposals_GT_labels[i])
            aligned_GT_labels_list.append(torch.full((num_neg,), self.background_label_id, dtype=torch.long, device=feature_map.device))

            # Track which indices are positive vs negative
            is_positive_list.append(torch.ones(num_pos, dtype=torch.bool, device=feature_map.device))
            is_positive_list.append(torch.zeros(num_neg, dtype=torch.bool, device=feature_map.device))
        
        # Run the detector
        class_logits, mixed_bbox_deltas = self.roi_and_forward_detector_training(roi_pool, upscale_factor, B, H, W, feature_map, proposals=mixed_proposals)
        
        aligned_GT_labels = torch.cat(aligned_GT_labels_list, dim=0)
        is_positive_mask = torch.cat(is_positive_list, dim=0)

        # Retrieve the bbox_deltas for positive proposals only using the boolean mask
        aligned_GT_labels_pos = torch.cat(aligned_positive_proposals_GT_labels, dim=0)
        bbox_deltas_map = mixed_bbox_deltas[is_positive_mask]

        aligned_GT_BB = torch.cat(aligned_positive_proposals_GT_BB, dim=0)
        positive_proposals_concat = torch.cat(positive_proposals, dim=0)

        # loss classification of detector (only positive proposals have a GT therefore no regression for negative proposals)
        if aligned_GT_labels.numel() == 0:
            loss_detector = (class_logits * 0).sum() # The aim is to have ~0 loss and not retropropagate None gradient but 0
        else:
            loss_detector = nn.functional.cross_entropy(
                class_logits, aligned_GT_labels.long()
            )

        λ_reg = 10
        loss_reg_fn = nn.SmoothL1Loss()
        if aligned_GT_BB.numel() > 0:
            # Calculate Target Deltas
            p_y, p_x, p_h, p_w = positive_proposals_concat[:, 0], positive_proposals_concat[:, 1], positive_proposals_concat[
                                                                                                   :,
                                                                                                   2], positive_proposals_concat[
                                                                                                       :, 3]

            # Ground-truth boxes for these proposals (g_y, g_x, g_h, g_w)
            g_y, g_x, g_h, g_w = aligned_GT_BB[:, 0], aligned_GT_BB[:, 1], aligned_GT_BB[:, 2], aligned_GT_BB[:, 3]

            # Calculate the center coordinates for both proposals and GT boxes
            p_center_y = p_y + 0.5 * p_h
            p_center_x = p_x + 0.5 * p_w
            g_center_y = g_y + 0.5 * g_h
            g_center_x = g_x + 0.5 * g_w

            # Target delta formulas (same as RPN, but using proposals as anchors)
            t_y = (g_center_y - p_center_y) / p_h
            t_x = (g_center_x - p_center_x) / p_w
            t_h = torch.log(g_h / p_h)
            t_w = torch.log(g_w / p_w)
            target_deltas = torch.stack((t_y, t_x, t_h, t_w), dim=1)

            # scaled target deltas to smooth between coordinates and length => to avoid exploding deltas predictions
            # => need to scale back in inference
            target_deltas = target_deltas / self.bbox_normalize_stds

            # Select the Predicted Deltas corresponding to the GT class
            indices_for_deltas = torch.arange(bbox_deltas_map.size(0), device=feature_map.device)

            # Select the deltas that correspond to the ground-truth class of each proposal
            predicted_deltas = bbox_deltas_map.view(-1, self.n_class + 1, 4)[indices_for_deltas, aligned_GT_labels_pos]

            #print("detector class loss:", loss_detector.item())
            # Compute the loss between predicted and target deltas ---
            loss_reg_val = loss_reg_fn(predicted_deltas, target_deltas)
            #print("Detector reg loss:", loss_reg_val.item())
            loss_detector += loss_reg_val # λ_reg * loss_reg_val

        # final model loss
        loss = loss_rpn + loss_detector

        #print("Final model loss:", loss.item())
        return loss#, final_anchors_boxes
    

    def roi_and_forward_detector_inference(self, roi_pool, upscale_factor, batch_size, h, w,feature_map, objectness_prob_map, proposals):
        # apply filters (objectness threshold, NMS, top-k boxes) to select only the best bounding boxes
        ROIs = []

        for image in range(batch_size):

            # only keep positive bounding boxes
            mask_positive_boxes = objectness_prob_map[image] > self.objectness_iou_threshold_positive_boxes                
            positive_boxes_objectness_map = objectness_prob_map[image][mask_positive_boxes]
            positive_proposals = proposals[image][mask_positive_boxes]

            # Keep cutting if there are still too many boxes (for speed optimization)
            if positive_boxes_objectness_map.shape[0] > 2000:
                positive_boxes_objectness_map, indices = torch.topk(positive_boxes_objectness_map, 2000)
                positive_proposals = positive_proposals[indices]

            # clip the bounding box and transform h and w into x_max and y_max for torchvision NMS and ROI operations
            positive_proposals_formatted = clip_bb_and_transform(positive_proposals, h * upscale_factor[0], w * upscale_factor[1]).float()

            #apply NMS and keep the 500 most relevants bounding boxes
            keep_indices = ops.nms(positive_proposals_formatted, positive_boxes_objectness_map, iou_threshold=0.3)[:500]
            positive_proposals_formatted = positive_proposals_formatted[keep_indices]

            ROIs.append(positive_proposals_formatted)

        # ROI pooling : extract feature map block
        pooled_features = roi_pool(feature_map, ROIs)

        # detector part : classify the bounding boxes and refine them
        class_logits, bbox_deltas_map = self.detector.forward(pooled_features)
        class_probabilities = self.detector_softmax(class_logits)
        return class_probabilities, bbox_deltas_map, ROIs
    

    def adapt_proposals_format(self, proposals):
        """ Convert from (x_min, y_min, x_max, y_max) format to (center_y, center_x, h, w) format """
        x_min = proposals[:, 0]
        y_min = proposals[:, 1]
        x_max = proposals[:, 2]
        y_max = proposals[:, 3]

        w = x_max - x_min
        h = y_max - y_min

        cxp = x_min + 0.5 * w
        cyp = y_min + 0.5 * h
        return torch.stack([cyp, cxp, h, w], dim=-1)


    def forward(self, feature_map, upscale_factor=None):

        roi_pool, upscale_factor = self.ensure_parameters(upscale_factor)

        B, H, W, _ = feature_map.shape
        feature_map = feature_map.permute(0, 3, 1, 2).contiguous()

        # RPN part : create relevant bounding boxes from the feature maps
        preds = self.rpn.forward(feature_map, upscale_factor, train=False)
        objectness_logit_map, proposals, _, _ = preds

        # Transform logit score to probability score
        objectness_prob_map = self.rpn_objectness_sigmoid(objectness_logit_map)

        # Select the region of interest on the feature map (ROI) and forward the detector head to classify the proposals and compute deltas
        class_probabilities, bbox_deltas_map, ROIs = self.roi_and_forward_detector_inference(roi_pool, upscale_factor, B, H, W, feature_map, objectness_prob_map, proposals)

        # Track the most probable class for each proposal and the corresponding confidence score
        confidence_score, indice_class_selected = torch.max(class_probabilities, dim=-1)

        # Unformat the proposals to be able to apply the deltas to them and get the final refined bounding boxes.
        # The proposals are in (x_min, y_min, x_max, y_max) format, we need to convert them to (center_y, center_x, h, w) format.
        positive_proposals_formatted = torch.cat(ROIs, dim=0)
        positive_proposals_centered = self.adapt_proposals_format(positive_proposals_formatted)

        # We want to Un-normalize by multiplying with the stds values
        # The predicted deltas have shape (num_proposals, (n_class + 1) * 4)
        # The stds values have shape (4)
        # => Reshape the deltas to allow broadcast and then broadcast multiplication to unscaled the deltas
        num_proposals = bbox_deltas_map.shape[0]
        deltas_reshaped = bbox_deltas_map.view(num_proposals, (self.n_class + 1), 4)
        unscaled_deltas_reshaped = deltas_reshaped * self.bbox_normalize_stds

        # Select the deltas for the most probable class
        # First, create an index for the first dimension (proposals)
        proposal_indices = torch.arange(num_proposals, device=bbox_deltas_map.device)
        # Then use this indices along with the indice_class_selected to select the correct deltas for each proposal
        selected_deltas = unscaled_deltas_reshaped[proposal_indices, indice_class_selected]

        # Retrieve the coordinates
        cyp = positive_proposals_centered[:, 0]
        cxp = positive_proposals_centered[:, 1]
        hp = positive_proposals_centered[:, 2]
        wp = positive_proposals_centered[:, 3]

        dy = selected_deltas[:, 0]
        dx = selected_deltas[:, 1]
        dh = selected_deltas[:, 2]
        dw = selected_deltas[:, 3]

        # Apply the delta transformations
        pred_center_y = hp * dy + cyp
        pred_center_x = wp * dx + cxp

        # Clamping dh and dw to prevent overflow
        dh = torch.clamp(dh, max=math.log(1000. / 16.))
        dw = torch.clamp(dw, max=math.log(1000. / 16.)) 

        pred_h = hp * torch.exp(dh)
        pred_w = wp * torch.exp(dw)

        # shift back to (x_min, y_min)
        x_new = pred_center_x - 0.5 * pred_w
        y_new = pred_center_y - 0.5 * pred_h

        # The final refined boxes (in center format)
        if self.xywh_format:
            final_boxes = torch.stack([x_new, y_new, pred_w, pred_h], dim=-1)
        else:
            final_boxes = torch.stack([y_new, x_new, pred_h, pred_w], dim=-1)

        return confidence_score, ROIs, final_boxes, indice_class_selected
    

    @staticmethod
    def postprocess_remove_low_confidence_boxes(confidence_score_threshold, confidence_score, ROIs, final_boxes, indice_class_selected):
        """ Remove boxes with low confidence scores """

        mask = confidence_score > confidence_score_threshold
        confidence_score = confidence_score[mask]
        final_boxes = final_boxes[mask]
        indice_class_selected = indice_class_selected[mask]

        new_ROIs = []
        for ROI, mask in zip(ROIs, mask.split([len(ROI) for ROI in ROIs])):
            new_ROIs.append(ROI[mask])

        return confidence_score, new_ROIs, final_boxes, indice_class_selected
    

    @staticmethod
    def postprocess_match_label(label, confidence_score, ROIs, final_boxes, indice_class_selected):
        """ Keep bounding boxes corresponding to desired label """
        mask = indice_class_selected == label
        confidence_score = confidence_score[mask]
        final_boxes = final_boxes[mask]
        indice_class_selected = indice_class_selected[mask]

        new_ROIs = []
        for ROI, mask in zip(ROIs, mask.split([len(ROI) for ROI in ROIs])):
            new_ROIs.append(ROI[mask])

        return confidence_score, new_ROIs, final_boxes, indice_class_selected
    
    @staticmethod
    def postprocess_exclude_label(label, confidence_score, ROIs, final_boxes, indice_class_selected):
        """ Exclude bounding boxes corresponding to desired label """
        mask = indice_class_selected == label
        confidence_score = confidence_score[~mask]
        final_boxes = final_boxes[~mask]
        indice_class_selected = indice_class_selected[~mask]

        new_ROIs = []
        for ROI, mask in zip(ROIs, (~mask).split([len(ROI) for ROI in ROIs])):
            new_ROIs.append(ROI[mask])

        return confidence_score, new_ROIs, final_boxes, indice_class_selected
    

    @staticmethod
    def postprocess_gather_by_image(confidence_score, ROIs, final_boxes, indice_class_selected):
        """ Gather the predicted boxes and labels by image in the batch (instead of concatenated) """

        split_sizes = [len(ROI) for ROI in ROIs]

        pred_boxes = torch.split(final_boxes, split_sizes, dim=0)
        labels = torch.split(indice_class_selected, split_sizes, dim=0)
        confidence_scores = torch.split(confidence_score, split_sizes, dim=0)

        return confidence_scores, pred_boxes, labels



if __name__ == '__main__':

    in_channels = 1
    batch_size, h, w = 2, 50, 50
    upscale_factor = 16, 16
    num_anchors = 9
    n_class = 2
    labels = torch.tensor([[1],[2]], dtype=torch.long)
    #labels_with_update = torch.tensor([[0,-1],[2,2]], dtype=torch.long)

    feature_map = torch.arange(batch_size * in_channels * h * w).view([batch_size, in_channels, h, w]).float().expand(2, in_channels,h,w)

    box22 = torch.tensor([[[160,160,128,128]],[[400,400,200,200]]])
    #box22_with_update = torch.tensor([[[160,160,128,128],[-1,-1,-1,-1]],[[400,400,200,200],[400,400,200,200]]])


    #box11 = torch.arange(2*h*w*9*4).reshape([2,h*w*9,4])
    #box22 = torch.stack([torch.tensor(160),torch.tensor(160),torch.tensor(128),torch.tensor(128)], dim=-1).unsqueeze(0).unsqueeze(0).expand(2,1,4)

    #preds = box11[...,0]/(200000), box11
    #loss_rpn, positive_anchors, aligned_GT_BB, aligned_GT_labels = loss_RPN(preds, box22, labels)


    model = Faster_RCNN(in_channels, num_anchors, n_class, [h,w,upscale_factor])
    #print(model.forward_train(feature_map, box22, labels))

    class_probabilities, ROIs, final_boxes, indice_class_selected = model.forward(feature_map)

    print(len(class_probabilities), len(final_boxes), len(indice_class_selected))
    print(class_probabilities[0].shape, final_boxes[0].shape, indice_class_selected[0].shape)


    X = """

    print("with -1 mask : ")

    print(model.train(feature_map, box22_with_update, labels_with_update))
    
    #class_probabilities, ROIs, anchors_boxes = model(feature_map)
    #print(class_probabilities.shape, ROIs[0].shape, ROIs[1].shape, anchors_boxes[0].shape, anchors_boxes[1].shape)


    roi_pool = ops.RoIPool(output_size=(7, 7), spatial_scale=1/upscale_factor[0])

    #feature_map = torch.full([in_channels, h, w],1).reshape(1, in_channels, h, w).float()

    rpn = RPN(in_channels, num_anchors,[h,w,upscale_factor])

    feature_map = torch.arange(2 * in_channels * h * w).reshape([2, in_channels, h, w]).float()
    rpn.forward(feature_map, train=True)
    #anchor_objectness, anchors_coordinates = rpn(feature_map)

    #box1 = torch.stack([torch.tensor(0),torch.tensor(0),torch.tensor(10),torch.tensor(10)], dim=-1)
    #box2 = torch.stack([torch.tensor(-5),torch.tensor(-5),torch.tensor(10),torch.tensor(10)], dim=-1)


    box11 = torch.arange(2*h*w*9*4).reshape([2,h*w*9,4])
    box22 = torch.stack([torch.tensor(160),torch.tensor(160),torch.tensor(128),torch.tensor(128)], dim=-1).unsqueeze(0).unsqueeze(0).expand(2,1,4)

    preds = box11[...,0]/(200000), box11
    loss_rpn, positive_anchors, aligned_GT_BB = loss_RPN(preds, box22)

    ROIs = []
    loss = 0
    for image in range(batch_size):

        #clip the bounding box and transform h and w into x_max and y_max for torchvision ROI operation
        ROIs.append(clip_bb_and_transform(positive_anchors[image], h*upscale_factor[0], w*upscale_factor[1]).float())

    #extract feature map block
    pooled_features = roi_pool(feature_map, ROIs)

    detector = FastRCNN(1, 2, 24)

    class_probabilities, bbox_deltas_map = detector.forward(pooled_features)
    positive_anchors = torch.cat(positive_anchors, dim=0)
    aligned_GT_BB = torch.cat(aligned_GT_BB, dim=0)
    print(class_probabilities.shape, bbox_deltas_map.shape, positive_anchors.shape, aligned_GT_BB.shape)

    GT_class_probabilities = torch.full([4,3], 0)
    for i, label in enumerate(labels):
        GT_class_probabilities[i,label] = 1

    # loss classification
    loss += nn.functional.binary_cross_entropy(class_probabilities, GT_class_probabilities.float())

    # compute bounding box shift
    print(bbox_deltas_map)
    mask_corresponding_values = torch.arange(bbox_deltas_map.size(0)) * (num_classes + 1) * 4 + 4*torch.argmax(class_probabilities, dim=-1)
    bbox_deltas_map = bbox_deltas_map.reshape(-1)
    delta_x = bbox_deltas_map[mask_corresponding_values + 0]
    delta_y = bbox_deltas_map[mask_corresponding_values + 1]
    delta_h = bbox_deltas_map[mask_corresponding_values + 2]
    delta_w = bbox_deltas_map[mask_corresponding_values + 3]

    #print(positive_anchors[:,2].shape, delta_x.shape)
    x_new = positive_anchors[:, 2] * delta_x + positive_anchors[:, 0]
    y_new = positive_anchors[:, 3] * delta_y + positive_anchors[:, 1]
    h_new = positive_anchors[:, 2] * torch.exp(delta_h)
    w_new = positive_anchors[:, 3] * torch.exp(delta_w)

    anchor_boxes = torch.stack([x_new, y_new, h_new, w_new], dim=-1)

    #compute regression loss
    loss_reg = nn.SmoothL1Loss()
    loss += 10 * loss_reg(anchor_boxes, aligned_GT_BB.float()) / anchor_boxes.shape[0]





    box1 = torch.arange(2*h*w*9*4).reshape([2,h,w,9,4])
    print("deuxieme essaie")
    loss_RPN(preds, box1[0,0])
    """
