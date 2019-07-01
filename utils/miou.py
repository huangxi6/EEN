import numpy as np
import cv2
import os
import json
from collections import OrderedDict
import argparse
from PIL import Image as PILImage
from utils.transforms import transform_parsing

LABELS_LIP = ['Background', 'Hat', 'Hair', 'Glove', 'Sunglasses', 'Upper-clothes', 'Dress', 'Coat', \
          'Socks', 'Pants', 'Jumpsuits', 'Scarf', 'Skirt', 'Face', 'Left-arm', 'Right-arm', 'Left-leg',
          'Right-leg', 'Left-shoe', 'Right-shoe']
LABELS_PERSON = ['Background', 'head', 'torso', 'u-arms', 'l-arms', 'u-legs', 'l-legs']
LABELS_CIHP= ['Background', 'Hat', 'Hair', 'Glove', 'Sunglasses', 'Upper-clothes', 'Dress', 'Coat', \
          'Socks', 'Pants', 'Torso-skin', 'Scarf', 'Skirt', 'Face', 'Left-arm', 'Right-arm', 'Left-leg',
          'Right-leg', 'Left-shoe', 'Right-shoe']

def get_palette(num_cls):
    """ Returns the color map for visualizing the segmentation mask.
    Args:
        num_cls: Number of classes
    Returns:
        The color map
    """

    n = num_cls
    palette = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        while lab:
            palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i += 1
            lab >>= 3
    return palette

def get_lip_palette():  
    palette = [ 0,0,0,
          128,0,0,
          255,0,0,
          0,85,0,
          170,0,51,
          255,85,0,
          0,0,85,
          0,119,221,
          85,85,0,
          0,85,85,
          85,51,0,
          52,86,128,
          0,128,0,
          0,0,255,
          51,170,221,
          0,255,255,
          85,255,170,
          170,255,85,
          255,255,0,
          255,170,0] 
    return palette

def get_confusion_matrix(gt_label, pred_label, num_classes):
    """
    Calcute the confusion matrix by given label and pred
    :param gt_label: the ground truth label
    :param pred_label: the pred label
    :param num_classes: the nunber of class
    :return: the confusion matrix
    """
    index = (gt_label * num_classes + pred_label).astype('int32')
    label_count = np.bincount(index)
    confusion_matrix = np.zeros((num_classes, num_classes))

    for i_label in range(num_classes):
        for i_pred_label in range(num_classes):
            cur_index = i_label * num_classes + i_pred_label
            if cur_index < len(label_count):
                confusion_matrix[i_label, i_pred_label] = label_count[cur_index]

    return confusion_matrix


def compute_mean_ioU(preds, scales, centers, num_classes, datadir, input_size=[473, 473], dataset='val'):
    list_path = os.path.join(datadir, dataset + '_id.txt')
    val_id = [i_id.strip() for i_id in open(list_path)]

    confusion_matrix = np.zeros((num_classes, num_classes))
    palette = get_lip_palette()

    for i, im_name in enumerate(val_id):
        gt_path = os.path.join(datadir, dataset + '_segmentations', im_name + '.png')
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        h, w = gt.shape
        pred_out = preds[i]
        s = scales[i]
        c = centers[i]
        pred = transform_parsing(pred_out, c, s, w, h, input_size)
        ################### Save the presiction results
        """output_im = PILImage.fromarray(np.asarray(pred, dtype=np.uint8))
        output_im.putpalette(palette)
        output_im.save('./output_cihp_3edge/'+im_name+'.png')"""
        ###################

        gt = np.asarray(gt, dtype=np.int32)
        pred = np.asarray(pred, dtype=np.int32)

        ignore_index = gt != 255

        gt = gt[ignore_index]
        pred = pred[ignore_index]

        confusion_matrix += get_confusion_matrix(gt, pred, num_classes)

    pos = confusion_matrix.sum(1)
    res = confusion_matrix.sum(0)
    tp = np.diag(confusion_matrix)

    pixel_accuracy = (tp.sum() / pos.sum()) * 100
    mean_accuracy = ((tp / np.maximum(1.0, pos)).mean()) * 100
    IoU_array = (tp / np.maximum(1.0, pos + res - tp))
    IoU_array = IoU_array * 100
    #print(IoU_array)
    mean_IoU = IoU_array.mean()
    print('Pixel accuracy: %f \n' % pixel_accuracy)
    print('Mean accuracy: %f \n' % mean_accuracy)
    print('Mean IU: %f \n' % mean_IoU)
    name_value = []

    for i, (label, iou) in enumerate(zip(LABELS_LIP, IoU_array)):
        name_value.append((label, iou))

    name_value.append(('Pixel accuracy', pixel_accuracy))
    name_value.append(('Mean accuracy', mean_accuracy))
    name_value.append(('Mean IU', mean_IoU))
    name_value = OrderedDict(name_value)
    return name_value

def compute_mean_ioU_file(preds_dir, num_classes, datadir, dataset='val'):
    list_path = os.path.join(datadir, dataset + '_id.txt')
    val_id = [i_id.strip() for i_id in open(list_path)]

    confusion_matrix = np.zeros((num_classes, num_classes))

    for i, im_name in enumerate(val_id):
        gt_path = os.path.join(datadir, dataset + '_segmentations', im_name + '.png')
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)

        pred_path = os.path.join(preds_dir, im_name + '.png')
        pred = np.asarray(PILImage.open(pred_path))

        gt = np.asarray(gt, dtype=np.int32)
        pred = np.asarray(pred, dtype=np.int32)

        ignore_index = gt != 255

        gt = gt[ignore_index]
        pred = pred[ignore_index]

        confusion_matrix += get_confusion_matrix(gt, pred, num_classes)

    pos = confusion_matrix.sum(1)
    res = confusion_matrix.sum(0)
    tp = np.diag(confusion_matrix)

    pixel_accuracy = (tp.sum() / pos.sum())*100
    mean_accuracy = ((tp / np.maximum(1.0, pos)).mean())*100
    IoU_array = (tp / np.maximum(1.0, pos + res - tp))
    IoU_array = IoU_array*100
    mean_IoU = IoU_array.mean()
    print('Pixel accuracy: %f \n' % pixel_accuracy)
    print('Mean accuracy: %f \n' % mean_accuracy)
    print('Mean IU: %f \n' % mean_IoU)
    name_value = []

    for i, (label, iou) in enumerate(zip(LABELS_LIP, IoU_array)):
        name_value.append((label, iou))

    name_value.append(('Pixel accuracy', pixel_accuracy))
    name_value.append(('Mean accuracy', mean_accuracy))
    name_value.append(('Mean IU', mean_IoU))
    name_value = OrderedDict(name_value)
    return name_value

def write_results(preds, scales, centers, datadir, dataset, result_dir, input_size=[473, 473]):
    palette = get_palette(20)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    json_file = os.path.join(datadir, 'annotations', dataset + '.json')
    with open(json_file) as data_file:
        data_list = json.load(data_file)
        data_list = data_list['root']
    for item, pred_out, s, c in zip(data_list, preds, scales, centers):
        im_name = item['im_name']
        w = item['img_width']
        h = item['img_height']
        pred = transform_parsing(pred_out, c, s, w, h, input_size)
        #pred = pred_out
        save_path = os.path.join(result_dir, im_name[:-4]+'.png')

        output_im = PILImage.fromarray(np.asarray(pred, dtype=np.uint8))
        output_im.putpalette(palette)
        output_im.save(save_path)

def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLabLFOV NetworkEv")
    parser.add_argument("--pred-path", type=str, default='',
                        help="Path to predicted segmentation.")
    parser.add_argument("--gt-path", type=str, default='',
                        help="Path to the groundtruth dir.")

    return parser.parse_args()


if __name__ == "__main__":
    args = get_arguments()
    palette = get_palette(20)
    
    pred_dir = './snapshots/'
    num_classes = 20
    datadir = './dataset/LIP'
    compute_mean_ioU_file(pred_dir, num_classes, datadir, dataset='val')