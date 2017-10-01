from PIL import Image

files = []
for i in range(1, 925):
    i = str(i)
    i = list(i)
    z = 5 - len(i)
    for k in range(z):
        i.insert(0,'0')
    i = ''.join(i)
    files.append(i)

for i in range(924):
    try:
        im = Image.open(files[i] + ".jpg")
        im.save(files[i] + ".ppm")
    except:
        continue
