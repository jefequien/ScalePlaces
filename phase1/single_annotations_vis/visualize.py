import sys
import os

def generateHTML(folder_path):
    folder_name = os.path.basename(os.path.normpath(folder_path))

    max_num = 20
    body = "<body>"
    for p in os.listdir(folder_path):
        percentile_dir = os.path.join(folder_path, p)
        num = 0
        body += "<br>{}<br>".format(p)
        for filename in os.listdir(percentile_dir):
            if ".jpg" in filename or ".png" in filename:
                image_path = os.path.join(percentile_dir, filename)
                image_tag = "<img src=\"{}\" height=\"256px\">".format(image_path)
                num += 1
                if num == max_num:
                    break

    body += "</body>"

    html = "<html><head></head>{}</html>".format(body)

    output_file = "{}.html".format(folder_name)
    with open(output_file, 'w') as f:
        f.write(html)

    print "http://places.csail.mit.edu/scaleplaces/ScalePlaces/phase1/single_annotations_vis/{}".format(output_file)


folder_path = sys.argv[1]
generateHTML(folder_path)

# for im_folder in os.listdir(folder_path):
#     im_folder_path = os.path.join(folder_path, im_folder)
#     if os.path.isdir(im_folder_path):
#         generateHTML(im_folder_path)

