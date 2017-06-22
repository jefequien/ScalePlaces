import os

from scipy import misc
import utils


project = "ade20k"
config = utils.get_data_config(project)
root_images = config["images"]
root_cm = os.path.join(config["pspnet_prediction"], "category_mask")
root_pm = os.path.join(config["pspnet_prediction"], "prob_mask")
root_gt = config["ground_truth"]

im_list = [line for line in open(config["im_list"], 'r')]

def evaluate_image(im):
    cm = misc.imread(os.path.join(root_cm, im.replace(".jpg",".png")))
    gt = misc.imread(os.path.join(root_gt, im.replace(".jpg",".png")))

    results = {}
    for c in xrange(1,151):
        cm_mask = cm == c
        gt_mask = gt == c
        intersection = np.ma.mask_and(cm_mask, gt_mask)
        union = np.ma.mask_or(cm_mask, gt_mask)

        if np.sum(union) != 0:
            iou = np.sum(intersection)/np.sum(union)
            gt_area = np.sum(gt_mask)
            results[c] = (iou, gt_area)
    return results

def evaluate_categories():
    accuracies = {}
    for im in im_list:
        results = evaluate_image(im)
        for c in results:
            acc, area = results[c]
            if c not in accuracies:
                accuracies[c] = [acc]
            else:
                accuracies[c].append(acc)


    output = "0 CATEGORY AVG_ACC NUM\n"
    categories = utils.get_categories()
    for c in xrange(1,151):
        accs = accuracies[c]
        output += "{} {} {} {}\n".format(c, categories[c], sum(accs)/len(accs), len(accs))
    with open("baseline.txt", 'w') as f:
        f.write(output)

evaluate_categories()

