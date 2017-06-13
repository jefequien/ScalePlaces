import sys
import os

folder_path = sys.argv[1]
folder_name = os.path.basename(os.path.normpath(folder_path))

num = 0
max_num = 1000
body = "<body>"
for filename in os.listdir(folder_path):
    if ".jpg" in filename or ".png" in filename:
        image_path = os.path.join(folder_path, filename)
        image_tag = "<img src=\"{}\" width=\"256px\">".format(image_path)
        body += "<br><br> {} <br> {}".format(image_path, image_tag)
        num += 1
        if num == max_num:
            body += "</body>"
            break

html = "<html><head></head>{}</html>".format(body)

with open("{}.html".format(folder_name), 'w') as f:
    f.write(html)