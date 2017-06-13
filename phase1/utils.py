
def get_categories():
    categories = {}
    with open("objectInfo150.txt", 'r') as f:
        for line in f.readlines():
            split = line.split()
            cat = split[0]
            if cat.isdigit():
                categories[int(cat)] = split[4].replace(',','')
        return categories

# categories = get_categories()
# print categories[1]

# with open("1wall.txt", 'w') as f:
#     f.write("hi")