from typing import Optional, Tuple

import torch
from torch import nn
import torchvision.ops as ops
import torch.nn.functional as F
from utils import loss_RPN, clip_bb_and_transform
from Fast_RCNN import FastRCNN

class RPN(nn.Module):
    def __init__(self, in_channels: int, num_anchors: int, bbox_normalize_stds: torch.Tensor, fixed_shape: Optional[Tuple[int, int, Tuple[int, int]]]=None, scales=[43,86,126,172,256], shapes=[[1,1],[1,2],[2,1]]):
        super(RPN, self).__init__()

        self.register_buffer("bbox_normalize_stds", bbox_normalize_stds)

        self.num_anchors = num_anchors
        self.scales =  scales
        self.shapes = shapes

        self.rpn_conv = nn.Conv2d(in_channels, 256, kernel_size=(3,3), stride=1, padding='same')
        #self.rpn_bn = nn.GroupNorm(num_groups=32, num_channels=256) #nn.BatchNorm2d(256)
        self.rpn_activation = nn.ReLU()

        self.rpn_objectness = nn.Conv2d(256, num_anchors, kernel_size=(1,1), stride=1, padding='same')

        self.rpn_bbox_pred = nn.Conv2d(256, num_anchors * 4, kernel_size=(1,1), stride=1, padding='same')

        # Initialize weights
        # for layer in [self.rpn_conv, self.rpn_objectness, self.rpn_bbox_pred]:
        #     torch.nn.init.normal_(layer.weight, mean=0.0, std=0.01)
        #     torch.nn.init.zeros_(layer.bias)

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

            grid = torch.stack(torch.meshgrid(position_H, position_W), dim=-1)

            # for each position: create N anchors
            anchors = []
            for scale in self.scales:
                for shape in self.shapes:
                    shape_tensor = torch.tensor(shape)
                    length = torch.tensor(scale * shape_tensor)

                    #shape : H, W, 4 : x_min, y_min, h, w
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

            x_min, y_min, h, w = anchors[..., 0], anchors[..., 1], anchors[..., 2], anchors[..., 3]
            x_max = x_min + h
            y_max = y_min + w

            valid_mask = (x_min >= 0) & (y_min >= 0) & (x_max <= height) & (y_max <= width)

            return anchors.view(-1, 4), valid_mask.view(-1)


    def forward(self, input, upscale_factor=None, train = False):
        B, H, W, _ = input.shape
        x = input.permute(0, 3, 1, 2).contiguous() #shape : B, C, H, W

        # If all inputs are of the same size : only compute reference anchors one time
        if self.fixed_shape:
            anchor_boxes = self.anchors_boxes
            if train == True :
                valid_mask = self.valid_mask
                anchor_boxes = anchor_boxes[valid_mask]
            else :
                valid_mask = torch.full([H * W * self.num_anchors], True, dtype=torch.bool)
        elif upscale_factor :
            if train == True :
                anchor_boxes, valid_mask = self.create_anchors_boxes(H, W, upscale_factor)
                anchor_boxes = anchor_boxes[valid_mask]
            else :
                anchor_boxes, _ = self.create_anchors_boxes(H, W, upscale_factor)
                valid_mask = torch.full([H * W * self.num_anchors], True, dtype=torch.bool)
            anchor_boxes = anchor_boxes.to(input.dtype)
            valid_mask = valid_mask.to(input.dtype)
        else :
            raise NotImplementedError(" please provide the upscale factor (ratio image_shape/feature_map_shape)")

        # one convolution layer
        x = self.rpn_conv(input)
        #x = self.rpn_bn(x)
        x = self.rpn_activation(x)

        # compute the objectness for each anchors and keep only the valid ones (valid == within the image range)
        objectness_list = self.rpn_objectness(x) # output shape: B, num_anchors, H, W
        objectness_score_map = objectness_list.permute(0, 2, 3, 1).contiguous().view(B, -1)[:, valid_mask] # OK

        # compute the bounding boxes shifting values for each anchors and keep only the valid ones (valid == within the image range)
        bbox_list = self.rpn_bbox_pred(x) # output shape: B, num_anchors*4, H, W
        box_deltas_map = bbox_list.permute(0, 2, 3, 1).contiguous().view(B, -1, 4)[:, valid_mask] # OK

        # un-normalize the deltas, normalized during training
        box_deltas_map_unscaled = box_deltas_map * self.bbox_normalize_stds

        anchor_x_min = anchor_boxes[..., 1]
        anchor_y_min = anchor_boxes[..., 0]
        anchor_h = anchor_boxes[..., 2]
        anchor_w = anchor_boxes[..., 3]

        # Convert anchors to (center_x, center_y, w, h)
        anchor_center_x = anchor_x_min + 0.5 * anchor_w
        anchor_center_y = anchor_y_min + 0.5 * anchor_h

        # The deltas predicted by the network
        dx = box_deltas_map_unscaled[..., 0]
        dy = box_deltas_map_unscaled[..., 1]
        dw = box_deltas_map_unscaled[..., 2]
        dh = box_deltas_map_unscaled[..., 3]

        # Apply the deltas to the anchor boxes
        pred_center_x = dx * anchor_w + anchor_center_x
        pred_center_y = dy * anchor_h + anchor_center_y
        pred_w = anchor_w * torch.exp(dw)
        pred_h = anchor_h * torch.exp(dh)

        # Convert the refined boxes back to (x_min, y_min, w, h)
        x_new = pred_center_x - 0.5 * pred_w
        y_new = pred_center_y - 0.5 * pred_h
        refined_boxes = torch.stack([x_new, y_new, pred_w, pred_h], dim=-1)

        return objectness_score_map, refined_boxes, box_deltas_map_unscaled, anchor_boxes


class Faster_RCNN(nn.Module):
    def __init__(self, in_channels: int, num_anchors: int, n_class: int, fixed_shape: Optional[Tuple[int, int, Tuple[int, int]]] = None, objectness_iou_threshold_positive_boxes: float = 0.7, objectness_iou_threshold_negative_boxes: float = 0.3):
        super(Faster_RCNN, self).__init__()

        self.background_label_id = 0
        self.objectness_iou_threshold_positive_boxes = objectness_iou_threshold_positive_boxes
        self.objectness_iou_threshold_negative_boxes = objectness_iou_threshold_negative_boxes

        if fixed_shape is None:
            self.fixed_shape = False
        else:
            self.upscale_factor = fixed_shape[2]
        self.n_class = n_class

        bbox_normalize_stds = torch.tensor([0.1, 0.1, 0.2, 0.2])
        self.register_buffer("bbox_normalize_stds", bbox_normalize_stds)
        self.rpn = RPN(in_channels, num_anchors, fixed_shape)
        self.rpn_objectness_sigmoid = nn.Sigmoid()

        self.roi_pool = ops.RoIPool(output_size=(7, 7), spatial_scale=1 / self.upscale_factor[0])
        self.detector = FastRCNN(in_channels, n_class, hidden_dim=1024, roi_output_size=(7, 7))
        self.detector_softmax = nn.Softmax(dim=-1)

    def roi_and_forward_detector(self, batch_size, H, W, feature_map, proposales):
        # Clip the bounding box and transform h and w into x_max and y_max for torchvision ROI operation
        ROIs = []
        for image in range(batch_size):
            ROIs.append(clip_bb_and_transform(proposales[image], H * self.upscale_factor[0], W * self.upscale_factor[1]).float())

        # ROI pooling : extract feature map block
        pooled_features = self.roi_pool(feature_map.float(), ROIs)

        # Detector head part : classify the bounding boxes and refine them
        class_logits, bbox_deltas_map = self.detector.forward(pooled_features)

        return class_logits, bbox_deltas_map


    def train(self, feature_map, GT_BB, labels):
        B, H, W, _ = feature_map.shape

        # RPN part : create relevant proposals from the feature maps
        preds = self.rpn.forward(feature_map, self.upscale_factor, train=True)
        loss_rpn, positive_proposals, negative_proposals, aligned_positive_proposals_GT_BB, aligned_positive_proposals_GT_labels = loss_RPN(preds, GT_BB, labels, self.bbox_normalize_stds, objectness_iou_threshold_positive=self.objectness_iou_threshold_positive_boxes, objectness_iou_threshold_negative=self.objectness_iou_threshold_negative_boxes)

        # Proposal part:
        # NEW METHOD
        ####
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
        class_logits, mixed_bbox_deltas = self.roi_and_forward_detector(B, H, W, feature_map, proposals=mixed_proposals)
        
        aligned_GT_labels = torch.cat(aligned_GT_labels_list, dim=0)
        is_positive_mask = torch.cat(is_positive_list, dim=0)

        # Retrieve the bbox_deltas for positive proposals only using the boolean mask
        bbox_deltas_map = mixed_bbox_deltas[is_positive_mask]

        aligned_GT_BB = torch.cat(aligned_positive_proposals_GT_BB, dim=0)
        positive_proposals_concat = torch.cat(positive_proposals, dim=0)
        ####

        # OLD METHOD 
        ####
        # positive_class_probabilities, bbox_deltas_map = self.roi_and_forward_detector(B, H, W, feature_map, proposals=positive_proposals)
        # negative_class_probabilities, _ = self.roi_and_forward_detector(B, H, W, feature_map, proposals=negative_proposals)
        # class_logits = torch.cat((positive_class_logits, negative_class_logits), dim=0)
        
        # # Format the proposals for later use in the loss of the regression part of the detector
        # positive_proposals_concat = torch.cat(positive_proposals, dim=0)
        # negative_proposals_concat = torch.cat(negative_proposals, dim=0)

        # # Format the labels to compute the loss of classification part of the detector
        # aligned_GT_BB = torch.cat(aligned_positive_proposals_GT_BB, dim=0)
        # aligned_positive_proposals_GT_labels = torch.cat(aligned_positive_proposals_GT_labels, dim=0)
        # aligned_negative_proposals_GT_labels = torch.full((negative_proposals_concat.shape[0],), self.background_label_id, device=feature_map.device, dtype=torch.long) # all negative proposals are background class
        # aligned_GT_labels = torch.cat((aligned_positive_proposals_GT_labels, aligned_negative_proposals_GT_labels), dim=0)
        # ####


        # loss classification of detector (only positive proposals have a GT therefore no regression for negative proposals)
        if aligned_GT_labels.numel() == 0:
            loss_detector = (class_logits * 0).sum() + 0.01 # The aim is to have ~0 loss and not retropropagate None gradient but 0
        else:
            loss_detector = nn.functional.cross_entropy(
                class_logits, aligned_GT_labels.long()
            )

        λ_reg = 10
        loss_reg_fn = nn.SmoothL1Loss()
        if aligned_GT_BB.numel() > 0:
            # Calculate Target Deltas
            p_x, p_y, p_w, p_h = positive_proposals_concat[:, 0], positive_proposals_concat[:, 1], positive_proposals_concat[
                                                                                                   :,
                                                                                                   2], positive_proposals_concat[
                                                                                                       :, 3]

            # Ground-truth boxes for these proposals (g_x, g_y, g_w, g_h)
            g_x, g_y, g_w, g_h = aligned_GT_BB[:, 0], aligned_GT_BB[:, 1], aligned_GT_BB[:, 2], aligned_GT_BB[:, 3]

            # Target delta formulas (same as RPN, but using proposals as anchors)
            t_x = (g_x - p_x) / p_w
            t_y = (g_y - p_y) / p_h
            t_w = torch.log(g_w / p_w)
            t_h = torch.log(g_h / p_h)
            target_deltas = torch.stack((t_x, t_y, t_w, t_h), dim=1)

            # scaled target deltas to smooth between coordinates and length => to avoid exploding deltas predictions
            # => need to scale back in inference
            target_deltas = target_deltas / self.bbox_normalize_stds

            # Select the Predicted Deltas corresponding to the GT class
            indices_for_deltas = torch.arange(bbox_deltas_map.size(0), device=feature_map.device)

            # Select the deltas that correspond to the ground-truth class of each proposal
            predicted_deltas = bbox_deltas_map.view(-1, self.n_class + 1, 4)[indices_for_deltas, aligned_positive_proposals_GT_labels]

            # Compute the loss between predicted and target deltas ---
            loss_reg_val = loss_reg_fn(predicted_deltas, target_deltas)
            loss_detector += λ_reg * loss_reg_val


        # final model loss
        loss = loss_rpn + loss_detector


        # if we need to print the final anchors boxes :
        printed_loss = """
        # make select shifting values corresponding to the class predicted
        #mask_corresponding_values = torch.arange(bbox_deltas_map.size(0), device=feature_map.device) * (self.n_class + 1) * 4 + 4 * aligned_GT_labels
        #bbox_deltas_map = bbox_deltas_map.reshape(-1)
        delta_x = bbox_deltas_map[mask_corresponding_values + 0]
        delta_y = bbox_deltas_map[mask_corresponding_values + 1]
        delta_h = bbox_deltas_map[mask_corresponding_values + 2]
        delta_w = bbox_deltas_map[mask_corresponding_values + 3]
        x_new = positive_predictions_concat[:, 2] * delta_x + positive_predictions_concat[:, 0]
        y_new = positive_predictions_concat[:, 3] * delta_y + positive_predictions_concat[:, 1]
        h_new = positive_predictions_concat[:, 2] * torch.exp(delta_h)
        w_new = positive_predictions_concat[:, 3] * torch.exp(delta_w)
        final_anchors_boxes = torch.stack([x_new, y_new, h_new, w_new], dim=-1)
        #lengths = [len(final_anchors_boxes[i]) for i in range(B)]
        #lengths.insert(0, 0)
        """

        return loss#, final_anchors_boxes

    def forward(self, feature_map):

        B, h, w, _ = feature_map.shape


        # RPN part : create relevant bounding boxes from the feature maps
        preds = self.rpn.forward(feature_map, self.upscale_factor, train=False)
        objectness_logit_map, refined_boxes, _, _ = preds

        # Transform logit score to probability score
        objectness_prob_map = self.rpn_objectness_sigmoid(objectness_logit_map)

        # apply filters (objectness threshold, NMS, top-k boxes) to select only the best bounding boxes
        ROIs = []

        for image in range(B):

            # only keep positive bounding boxes
            mask_positive_boxes = objectness_prob_map[image] > self.objectness_iou_threshold_positive_boxes                
            positive_boxes_objectness_map = objectness_prob_map[image][mask_positive_boxes]
            positive_boxes_coordinates = refined_boxes[image][mask_positive_boxes]

            # Keep cutting if there are still too many boxes (for speed optimization)
            if positive_boxes_objectness_map.shape[0] > 2000:
                positive_boxes_objectness_map, indices = torch.topk(positive_boxes_objectness_map, 2000)
                positive_boxes_coordinates = positive_boxes_coordinates[indices]

            # clip the bounding box and transform h and w into x_max and y_max for torchvision NMS and ROI operations
            positive_boxes_coordinates = clip_bb_and_transform(positive_boxes_coordinates, h * self.upscale_factor[0], w * self.upscale_factor[1]).float()

            #apply NMS and keep the 500 most relevants bounding boxes
            keep_indices = ops.nms(positive_boxes_coordinates, positive_boxes_objectness_map, iou_threshold=0.7)[:500]
            positive_boxes_coordinates = positive_boxes_coordinates[keep_indices]

            ROIs.append(positive_boxes_coordinates)

        # ROI pooling : extract feature map block
        pooled_features = self.roi_pool(feature_map, ROIs)

        # detector part : classify the bounding boxes and refine them
        class_logits, bbox_deltas_map = self.detector.forward(pooled_features)
        class_probabilities = self.detector_softmax(class_logits)
        confidence_score, indice_class_selected = torch.max(class_probabilities, dim=-1)



        ########################################################################
        #anchors = torch.cat(ROIs, dim=0)

        # make select shifting values corresponding to the class predicted
        #mask_corresponding_values = torch.arange(bbox_deltas_map.size(0)) * (self.n_class + 1) * 4 + 4 * indice_class_selected
        #bbox_deltas_map = bbox_deltas_map.reshape(-1,4)

        # un-normalize the deltas, normalized during training
        #box_deltas_map_unscaled = bbox_deltas_map * self.rpn.bbox_normalize_stds.to(bbox_deltas_map.device)
        #box_deltas_map_unscaled.reshape(-1)



        #delta_x = box_deltas_map_unscaled[mask_corresponding_values + 0]
        #delta_y = box_deltas_map_unscaled[mask_corresponding_values + 1]
        #delta_h = box_deltas_map_unscaled[mask_corresponding_values + 2]
        #delta_w = box_deltas_map_unscaled[mask_corresponding_values + 3]

        # compute bounding box shift (refinement of bounding boxes)
        #x_new = anchors[:, 2] * delta_x + anchors[:, 0]
        #y_new = anchors[:, 3] * delta_y + anchors[:, 1]
        #h_new = anchors[:, 2] * torch.exp(delta_h)
        #w_new = anchors[:, 3] * torch.exp(delta_w)

        # reshape the prediction to output bounding box after RPN and detector per batch
        #final_boxes = torch.stack([x_new, y_new, h_new, w_new], dim=-1)
        #####################################################################################

        proposal_boxes = torch.cat(ROIs, dim=0)

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

        # Apply the deltas to the proposal boxes to get the final refined boxes
        # (This is the same logic as in the RPN, but using proposal_boxes instead of anchors)
        # Make sure proposal_boxes are in (center_x, center_y, w, h) format for this calculation.
        # If they are in (x_min, y_min, x_max, y_max), you need to convert them first.

        # Assuming (center_x, center_y, w, h) format for proposal_boxes
        xp = proposal_boxes[:, 0]
        yp = proposal_boxes[:, 1]
        wp = proposal_boxes[:, 2]
        hp = proposal_boxes[:, 3]

        dx = selected_deltas[:, 0]
        dy = selected_deltas[:, 1]
        dw = selected_deltas[:, 2]
        dh = selected_deltas[:, 3]

        # Apply the delta transformations
        pred_center_x = wp * dx + xp
        pred_center_y = hp * dy + yp
        pred_w = wp * torch.exp(dw)
        pred_h = hp * torch.exp(dh)

        # The final refined boxes (in center format)
        final_boxes = torch.stack([pred_center_x, pred_center_y, pred_w, pred_h], dim=-1)

        return confidence_score, ROIs, final_boxes, indice_class_selected



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
