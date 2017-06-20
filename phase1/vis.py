import argparse
import os
import uuid

import utils

def getImageTag(path):
    tmp_file = "images/{}.jpg".format(uuid.uuid4().hex)
    tmp_path = os.path.join("tmp", tmp_file)
    if not os.path.exists(os.path.dirname(tmp_path)):
        os.makedirs(os.path.dirname(tmp_path))
    os.symlink(path, tmp_path)
    return "<img src=\"{}\" height=\"256px\">".format(tmp_file)

def makeImageSection(project, im):
    html = "{} {}<br><br>".format(project, im)

    config = utils.get_data_config(project)
    image = os.path.join(config["images"], im)
    category_mask = os.path.join(config["category_mask"], im.replace('.jpg','.png'))

    paths = [image, category_mask]
    for path in paths:
        html += getImageTag(path)

    html += "<br><br>"
    return html

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", help="Project name")
    parser.add_argument("-i", help="Image name")
    args = parser.parse_args()

    im = args.i
    project = args.p

    image_section = makeImageSection(project, im)
    body = "<body>{}</body>".format(image_section)
    html = "<html><head></head>{}</html>".format(body)

    output_file = "tmp/{}".format(im.replace('/','-').replace('.jpg','.html'))
    with open(output_file, 'w') as f:
        f.write(html)

    print "http://places.csail.mit.edu/scaleplaces/ScalePlaces/phase1/{}".format(output_file)

