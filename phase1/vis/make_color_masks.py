import os

import utils_vis as utils
from vis_image import ImageVisualizer

project = "street_view"
output_dir = "color_mask"
config = utils.get_data_config(project)
im_list = utils.open_im_list(project)


vis = ImageVisualizer(project)

for im in xrange(im_list):
    print im
    cm, cm_path = self.get_category_mask(im)
    if cm is not None:
        cm_color, cm_color_path = self.add_color(cm)
        output_path = os.path.join(output_dir, im)

        if not os.path.exists(os.path.dirname(output_path)):
            os.makedirs(os.path.dirname(output_path))
        os.rename(cm_color_path, output_path)
    else:
        print "Skipping", im
