import torch
import torch.nn as nn

def focal_loss(inputs, targets, alpha=0.01, gamma=2.0):
    BCE = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    probs = torch.sigmoid(inputs)
    pt = targets * probs + (1 - targets) * (1 - probs)

    # apply class-balanced alpha
    alpha_t = alpha * (1 - targets) + (1 - alpha) * targets

    focal_term = (1 - pt) ** gamma
    loss = alpha_t * focal_term * BCE

    return loss.mean()


def clip_bb_and_transform(bb, H, W):
    """ 
        Clip the bounding boxes to be within the image boundaries and transform them from (y_min, x_min, h, w) format to (x_min, y_min, x_max, y_max) format for torchvision operations.
    """

    y_min = bb[:, 0]
    x_min = bb[:, 1]
    y_max = y_min + bb[:, 2]
    x_max = x_min + bb[:, 3]

    # Clip to image boundaries
    y_min = torch.clamp(y_min, min=0, max=H)
    x_min = torch.clamp(x_min, min=0, max=W)
    y_max = torch.clamp(y_max, min=0, max=H)
    x_max = torch.clamp(x_max, min=0, max=W)

    return torch.stack((x_min, y_min, x_max, y_max), dim=1)

def IoU(box1, box2):

    y1_min, x1_min = box1[..., 0], box1[..., 1]
    y1_max, x1_max = y1_min + box1[..., 2], x1_min + box1[..., 3]

    y2_min, x2_min = box2[..., 0], box2[..., 1]
    y2_max, x2_max = y2_min + box2[..., 2], x2_min + box2[..., 3]

    # Compute intersection
    y_min_inter = torch.maximum(y1_min, y2_min)
    x_min_inter = torch.maximum(x1_min, x2_min)
    y_max_inter = torch.minimum(y1_max, y2_max)
    x_max_inter = torch.minimum(x1_max, x2_max)


    # Compute width and height of the intersection
    inter_h = torch.clamp(y_max_inter - y_min_inter, min=0)
    inter_w = torch.clamp(x_max_inter - x_min_inter, min=0)

    inter_area = inter_w * inter_h

    # Compute IoU
    return inter_area / ((x1_max - x1_min) * (y1_max - y1_min) + (x2_max - x2_min) * (y2_max - y2_min) - inter_area + 1e-6)


def second_positive_anchors_condition(iou, device, mask, objectness_iou_threshold):
    indices_positive_boxes = torch.argwhere(iou >= objectness_iou_threshold)

    iou_values = torch.cat((indices_positive_boxes, iou[indices_positive_boxes[:, 0], indices_positive_boxes[:, 1]].unsqueeze(1)), dim=1)
    _, inverse_indices = torch.unique(iou_values[:, 0], return_inverse=True)

    dtype = iou_values.dtype

    # retrieve the max IoU for each anchors > objectness_iou_threshold
    result = torch.full_like(inverse_indices, 0.0, dtype=dtype, device=device)
    result.scatter_reduce_(0, inverse_indices, iou_values[:, 2], reduce="amax")
    mask_max_value = iou_values[:, 2] == result[inverse_indices]
    IoU_max_value = iou_values[mask_max_value]

    # select the min in case one anchors has the exact same IoU over mulitple GT BB
    remaining_inverse_indices = inverse_indices[mask_max_value]
    result = torch.full_like(remaining_inverse_indices, torch.inf, dtype=dtype, device=device)
    result.scatter_reduce_(0, remaining_inverse_indices, IoU_max_value[:, 1], reduce="amin")
    mask_max_value_unique = IoU_max_value[:, 1] == result[remaining_inverse_indices]
    iou_max_value_unique = IoU_max_value[mask_max_value_unique]

    # assign the corresponding max IoU GT bb to the anchors with > objectness_iou_threshold IoU
    mask[0, indices_positive_boxes[:, 0].unique()] = 2
    mask[1, indices_positive_boxes[:, 0].unique()] = iou_max_value_unique[:, 1]
    return mask

def loss_RPN(preds, GT_bounding_boxes, GT_class_probabilities, bbox_normalize_stds, objectness_iou_threshold_positive, objectness_iou_threshold_negative):

    device = preds[1].device

    # retrieve the predictions from RPN
    objectness_score_map, proposals, preds_delta, base_anchor_boxes  = preds

    # B => Batch size, BB => number of bounding boxes predicted
    B, BB, _ = proposals.shape

    # for all images : compute positives anchors and losses
    loss = 0
    positive_proposals = []
    negative_proposals = []
    aligned_positive_proposals_GT_BB = []
    aligned_positive_proposals_GT_labels = []
    loss_reg = nn.SmoothL1Loss(reduction='sum')
    for i in range(B):

        # create a mask on the real labels (e.g without batching padding)
        mask_padding = GT_class_probabilities[i] != -1

        # check if there are GT Bounding boxes
        if mask_padding.any() :
            GT_bb_temp = GT_bounding_boxes[i][mask_padding]

            # Compute IoU for each anchor boxes to the ground truth bounding boxes in order to get the anchors corresponding to labels.
            anchor_temp = base_anchor_boxes[:,None,:].expand(BB, GT_bb_temp.shape[0], 4)  
            GT_bb_temp_expanded = GT_bb_temp[None,:,:].expand(BB, GT_bb_temp.shape[0], 4)
            iou = IoU(anchor_temp, GT_bb_temp_expanded)

            mask = torch.full((2, BB), 0, device=device, dtype=torch.float32) # First row is the status of the anchor (negative, neutral, positive). Second row : the GT BB index for the positive anchors related to it

            # negative anchors condition
            max_iou_per_anchor, _ = torch.max(iou, dim=1)
            indices_negative_boxes = torch.argwhere(max_iou_per_anchor < objectness_iou_threshold_negative)[:, 0]
            mask[0,indices_negative_boxes] = -1

            # second positive anchors condition: if IoU of predicted bounding box > objectness_iou_threshold_positive with any GT BB => positive and track with which GT BB has the max IoU 
            mask = second_positive_anchors_condition(iou, device, mask, objectness_iou_threshold = objectness_iou_threshold_positive)

            # first positive anchors condition: for each GT BB, the anchor with the highest IoU is positive
            indice_maxes = torch.argmax(iou, dim=0)
            mask[0,indice_maxes] = 1
            mask[1, indice_maxes] = torch.arange(indice_maxes.shape[0], device=device, dtype=mask.dtype)


            # compute RPN loss for each batch
            # first part : classification of anchor
            mask_relevant_anchors = mask[0,:] != 0
            relevant_anchors_objectness = objectness_score_map[i][mask_relevant_anchors]
            relevant_anchors_objectness_ground_truth = (mask[0,:][mask_relevant_anchors] + 1) / 2
            loss += focal_loss(relevant_anchors_objectness, relevant_anchors_objectness_ground_truth) * 10
            #print("RPN class loss:", loss.item())

            # Second part : Bounding box regression
            mask_positive_anchors = mask[0, :] == 1
            num_positive = torch.sum(mask_positive_anchors)
            mask_negative_anchors = mask[0, :] == -1

            if num_positive > 0:
                # Get the base anchors that were selected as positive
                selected_base_anchors = base_anchor_boxes[mask_positive_anchors]

                # Get the ground truth boxes assigned to them
                gt_indices_for_positives = mask[1, mask_positive_anchors].long()
                assigned_gt_boxes = GT_bb_temp[gt_indices_for_positives]

                # Get the deltas PREDICTED by the network for these anchors
                predicted_deltas_for_positives = preds_delta[i, mask_positive_anchors]

                # Calculate the TARGET deltas (the ground truth for the regression)
                ya, xa, ha, wa = selected_base_anchors[:, 0], selected_base_anchors[:, 1], selected_base_anchors[:,
                                                                                           2], selected_base_anchors[:,
                                                                                               3]
                yg, xg, hg, wg = assigned_gt_boxes[:, 0], assigned_gt_boxes[:, 1], assigned_gt_boxes[:,
                                                                                   2], assigned_gt_boxes[:, 3]
                
                # Calculate the center coordinates for both anchors and GT boxes
                cy_a = ya + 0.5 * ha
                cx_a = xa + 0.5 * wa
                cy_g = yg + 0.5 * hg
                cx_g = xg + 0.5 * wg


                # Target delta formulas from the paper
                t_y = (cy_g - cy_a) / ha
                t_x = (cx_g - cx_a) / wa
                t_h = torch.log(hg / ha)
                t_w = torch.log(wg / wa)
                target_deltas = torch.stack((t_y, t_x, t_h, t_w), dim=1)

                # scaled target deltas to smooth between coordinates and length => to avoid exploding deltas predictions
                # => need to scale back in inference
                target_deltas = target_deltas / bbox_normalize_stds

                # Compute the loss between predicted deltas and target deltas
                # Normalize by number of anchors ~2400 in the paper, we use number of relevant anchors here for stability
                loss_r = loss_reg(predicted_deltas_for_positives, target_deltas)
                loss_r = 0.1 * (loss_r / (num_positive * 2))
                #print("RPN reg loss:", loss_r.item())
                loss += loss_r


            positive_proposals.append(proposals[i][mask_positive_anchors])

            neg_boxes_tensor = proposals[i][mask_negative_anchors]
            if neg_boxes_tensor.shape[0] > num_positive:
                indices = torch.randperm(neg_boxes_tensor.shape[0], device=device)[:num_positive]
                neg_boxes_tensor = neg_boxes_tensor[indices]
            negative_proposals.append(neg_boxes_tensor)

            gt_indices_for_positives = mask[1, mask_positive_anchors].long()
            aligned_positive_proposals_GT_BB.append(GT_bb_temp[gt_indices_for_positives])
            aligned_positive_proposals_GT_labels.append(GT_class_probabilities[i][gt_indices_for_positives])

        else :

            GT_objectness_score_map = torch.zeros_like(objectness_score_map[i])
            loss += focal_loss(objectness_score_map, GT_objectness_score_map)

            positive_proposals.append(torch.empty((0, 4), device=device, dtype=torch.long))
            negative_proposals.append(torch.empty((0, 4), device=device, dtype=torch.long))
            aligned_positive_proposals_GT_BB.append(torch.empty((0, 4), device=device, dtype=torch.long))
            aligned_positive_proposals_GT_labels.append(torch.empty((0,), device=device, dtype=torch.long))

    return loss / B, positive_proposals, negative_proposals, aligned_positive_proposals_GT_BB, aligned_positive_proposals_GT_labels
