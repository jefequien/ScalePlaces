import sys
import os

def generateHTML(folder_path):
    folder_name = os.path.basename(os.path.normpath(folder_path))

    num = 0
    max_num = 1000
    body = "<body>"
    for filename in os.listdir(folder_path):
        if ".jpg" in filename or ".png" in filename:
            image_path = os.path.join(folder_path, filename)
            image_tag = "<img src=\"{}\" height=\"256px\">".format(image_path)
            body += "<br><br> {} <br> {}".format(image_path, image_tag)
            num += 1
            if num == max_num:
                body += "</body>"
                break

    html = "<html><head></head>{}</html>".format(body)

    output_file = "{}.html".format(folder_name)
    with open(output_file, 'w') as f:
        f.write(html)

    print "http://places.csail.mit.edu/scaleplaces/ScalePlaces/phase1/vis/{}".format(output_file)


folder_path = sys.argv[1]
# generateHTML(folder_path)

for im_folder in os.listdir(folder_path):
    im_folder_path = os.path.join(folder_path, im_folder)
    if os.path.isdir(im_folder_path):
        generateHTML(im_folder_path)

