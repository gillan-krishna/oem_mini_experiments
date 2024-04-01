import torch


def _threshold(x, threshold=None):
    return (x > threshold).type(x.dtype) if threshold is not None else x


# def fscore(pr, gt, beta=1, eps=1e-7, threshold=None, num_classes=9):
#     # sourcery skip: inline-immediately-returned-variable
#     """Calculate F-score between ground truth and prediction
#     Args:
#         pr (torch.Tensor): predicted tensor
#         gt (torch.Tensor):  ground truth tensor
#         beta (float): positive constant
#         eps (float): epsilon to avoid zero division
#         threshold: threshold for outputs binarization
#     Returns:
#         float: F score
#     """

#     pr = _threshold(pr, threshold=threshold)
#     tp = torch.sum(gt * pr)
#     fp = torch.sum(pr) - tp
#     fn = torch.sum(gt) - tp
#     score = ((1 + beta**2) * tp + eps) / ((1 + beta**2) * tp + beta**2 * fn + fp + eps)
#     return score

def fscore(pr, gt, beta=1, eps=1e-7, threshold=None, num_classes=9):
    # sourcery skip: inline-immediately-returned-variable
    """Calculate F-score between ground truth and prediction
    Args:
        pr (torch.Tensor): predicted tensor
        gt (torch.Tensor):  ground truth tensor
        beta (float): positive constant
        eps (float): epsilon to avoid zero division
        threshold: threshold for outputs binarization
    Returns:
        float: F score
    """

    pr = _threshold(pr, threshold=threshold)
    f_scores = torch.zeros(num_classes)

    for c in range(num_classes):
        class_pr = pr[:, c]
        class_gt = gt[:, c]
        tp = torch.sum(class_gt * class_pr)
        fp = torch.sum(class_pr) - tp
        fn = torch.sum(class_gt) - tp
        class_fscore = ((1 + beta**2) * tp + eps) / ((1 + beta**2) * tp + beta**2 * fn + fp + eps)
        f_scores[c] = class_fscore
    return f_scores

def iou(pr, gt, eps=1e-7, threshold=None, num_classes=9):
    """Calculate Intersection over Union between ground truth and prediction
    Args:
        pr (torch.Tensor): predicted tensor
        gt (torch.Tensor):  ground truth tensor
        eps (float): epsilon to avoid zero division
        threshold: threshold for outputs binarization
    Returns:
        float: IoU (Jaccard) score
    """
    class_iou = torch.zeros(num_classes)

    for c in range(num_classes):
        pr_c = _threshold(pr[:, c], threshold=threshold)
        gt_c = gt[:, c]
        
        # Expand dimensions to ensure broadcasting works correctly
        pr_c = pr_c.unsqueeze(1)
        gt_c = gt_c.unsqueeze(1)
        
        intersection = torch.sum(gt_c * pr_c)
        union = torch.sum(gt_c) + torch.sum(pr_c) - intersection
        
        class_iou[c] = (intersection + eps) / (union + eps)

    return class_iou