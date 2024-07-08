"""
FILENAME: Evaluations.py
DESCRIPTION: Evalutation functions 
@author: Jian Zhong
"""


import torch


# calculate intersect of union (IOU)
def intersection_over_union(pred, target, bkg_val = 0):
    
    label_colors = torch.unique(target[target != bkg_val])
    nof_colors = len(label_colors)
    iou = torch.zeros((nof_colors,), dtype = float)

    # calculate IOU for each label color (class)
    for i_color in range(nof_colors):
        cur_color = label_colors[i_color]
        pred_mask = pred == cur_color
        target_mask = target == cur_color
        intersect = torch.logical_and(pred_mask, target_mask).to(float)
        union = torch.logical_or(pred_mask, target_mask).to(float)
        iou[i_color] = torch.sum(intersect)/torch.sum(union)
        
    return iou, label_colors


# calulate interect of union of an entire dataset
def mean_iou_over_dataset(model, src_dataloader, device = "cpu", bkg_val = 0):
    tot_ious = dict()
    tot_nof_samples = 0

    model.to(device)
    model.eval()
    with torch.no_grad():
        for i_batch, data in enumerate(src_dataloader):
            inputs = data[0]
            targets = data[1]

            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)

            nof_samples = targets.size(0)
            for i_sample in range(nof_samples):
                cur_output = outputs[i_sample]
                cur_pred = torch.argmax(cur_output, dim = 0)
                cur_target = targets[i_sample]
                cur_ious, cur_labels = intersection_over_union(cur_pred, cur_target, bkg_val = bkg_val)
                for i_label in range(len(cur_labels)):
                    cur_label = cur_labels[i_label].item()
                    tot_ious[cur_label] = tot_ious.get(cur_label, 0) + cur_ious[i_label].item()

            tot_nof_samples += nof_samples
        
    avg_ious = dict()
    for cur_label in tot_ious.keys():
        avg_ious[cur_label] = tot_ious[cur_label]/tot_nof_samples

    return avg_ious
