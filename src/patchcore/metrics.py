"""Anomaly metrics."""
import numpy as np
from sklearn import metrics
import pickle # For debugging & generating joint roc_pr curves
import logging

LOGGER = logging.getLogger(__name__)

# Alternative pixelwise metrics based on metrics from mad_seminar. 
def compute_pixelwise_metrics_new(anomaly_segmentations, ground_truth_masks, negative_masks):
    if isinstance(anomaly_segmentations, list):
        anomaly_segmentations = np.stack(anomaly_segmentations)
    if isinstance(ground_truth_masks, list):
        ground_truth_masks = np.stack(ground_truth_masks)
    if isinstance(negative_masks, list):
        negative_masks = np.stack(negative_masks)

    # From mad-seminar:
    # NOTE: This is assuming that the images are only anomalous! If normal images are in test data (with class=fastmrixi), this is not accurate. 
    negative_masks[negative_masks > 0.5] = 1
    negative_masks[negative_masks < 1] = 0
    tps, fns, fps = 0, 0, []
    test_metrics = {
        'TP': [],
        'FP': [],
        'Precision': [],
        'Recall': [],
        'F1': [],
    }
    metrics = {
      'TP': [],
      'FP': [],
      'Precision': [],
      'Recall': [],
      'F1': [],
    }
    for ano_map_i, pos_mask_i, neg_mask_i in zip(anomaly_segmentations, ground_truth_masks, negative_masks):
        x_pos = ano_map_i * pos_mask_i
        x_neg = ano_map_i * neg_mask_i
        res_anomaly = np.sum(x_pos)
        res_healthy = np.sum(x_neg)

        amount_anomaly = np.count_nonzero(x_pos)
        amount_mask = np.count_nonzero(pos_mask_i)
        tp = 1 if amount_anomaly > 0.1 * amount_mask else 0  # 10% overlap due to large bboxes e.g. for enlarged ventricles.
        tn = 1 if amount_mask == 0 else 0 
        tps += tp
        fn = 1 if tp == 0 else 0
        fns += fn
        fp = int(res_healthy / max(res_anomaly, 1))
        fps.append(fp)
        # Imagewise precision
        precision = tp / max((tp + fp), 1)
        test_metrics['TP'].append(tp)
        test_metrics['FP'].append(fp)
        test_metrics['Precision'].append(precision)
        test_metrics['Recall'].append(tp)
        test_metrics['F1'].append(2 * (precision * tp) / (precision + tp + 1e-8))

    for metric in test_metrics:
        print('{} mean: {} +/- {}'.format(metric, np.nanmean(test_metrics[metric]),
                                            np.nanstd(test_metrics[metric])))
        if metric == 'TP':
            print(f'TP: {np.sum(test_metrics[metric])} of {len(test_metrics[metric])} detected')
        if metric == 'FP':
            print(f'FP: {np.sum(test_metrics[metric])} missed')
        metrics[metric].append(test_metrics[metric])

    # # LOGGER.info("Saving compute_pixelwise_metrics_new parameters.")
    # # Saving variables to generate joint roc_pr curves (Calculated from anomaly_seg and GT, NOT new metrics)
    # with open("evaluated_results/anomaly_test/flipped_pixelwise_metrics/BEST_IM224_WR50_L2-3_P1_D1024-1024_PS-3_AN-1_S0_0_0/new-pixelwise-metrics.pkl", 'wb') as file:
    #     pickle.dump((anomaly_segmentations,ground_truth_masks,negative_masks, test_metrics,metrics), file)
    
    return metrics



def compute_imagewise_retrieval_metrics(
    anomaly_prediction_weights, anomaly_ground_truth_labels
):
    """
    Computes retrieval statistics (AUROC, FPR, TPR).

    Args:
        anomaly_prediction_weights: [np.array or list] [N] Assignment weights
                                    per image. Higher indicates higher
                                    probability of being an anomaly.
        anomaly_ground_truth_labels: [np.array or list] [N] Binary labels - 1
                                    if image is an anomaly, 0 if not.
    """
    fpr, tpr, thresholds = metrics.roc_curve(
        anomaly_ground_truth_labels, anomaly_prediction_weights
    )
    auroc = metrics.roc_auc_score(
        anomaly_ground_truth_labels, anomaly_prediction_weights
    )
    return {"auroc": auroc, "fpr": fpr, "tpr": tpr, "threshold": thresholds}


def compute_pixelwise_retrieval_metrics(anomaly_segmentations, ground_truth_masks):
    """
    Computes pixel-wise statistics (AUROC, FPR, TPR) for anomaly segmentations
    and ground truth segmentation masks.

    Args:
        anomaly_segmentations: [list of np.arrays or np.array] [NxHxW] Contains
                                generated segmentation masks.
        ground_truth_masks: [list of np.arrays or np.array] [NxHxW] Contains
                            predefined ground truth segmentation masks
    """
    if isinstance(anomaly_segmentations, list):
        anomaly_segmentations = np.stack(anomaly_segmentations)
    if isinstance(ground_truth_masks, list):
        ground_truth_masks = np.stack(ground_truth_masks)

    flat_anomaly_segmentations = anomaly_segmentations.ravel()
    flat_ground_truth_masks = ground_truth_masks.ravel()

    fpr, tpr, thresholds = metrics.roc_curve(
        flat_ground_truth_masks.astype(int), flat_anomaly_segmentations
    )
    auroc = metrics.roc_auc_score(
        flat_ground_truth_masks.astype(int), flat_anomaly_segmentations
    )

    precision, recall, thresholds = metrics.precision_recall_curve(
        flat_ground_truth_masks.astype(int), flat_anomaly_segmentations
    )
    F1_scores = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) != 0,
    )

    optimal_threshold = thresholds[np.argmax(F1_scores)]
    predictions = (flat_anomaly_segmentations >= optimal_threshold).astype(int)
    fpr_optim = np.mean(predictions > flat_ground_truth_masks)
    fnr_optim = np.mean(predictions < flat_ground_truth_masks)

    return {
        "auroc": auroc,
        "fpr": fpr,
        "tpr": tpr,
        "optimal_threshold": optimal_threshold,
        "optimal_fpr": fpr_optim,
        "optimal_fnr": fnr_optim,
        "precision" : precision,
        "recall" : recall,
        "thresholds" : thresholds
    }
