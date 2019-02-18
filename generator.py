import xml.etree.ElementTree as ET
import os
import csv
from PIL    import Image
import string

CLASSES = string.digits + string.ascii_uppercase + string.ascii_lowercase
rootdir='English'
dataset_dir='kaist'

if not os.path.isdir(dataset_dir):
    os.makedirs(dataset_dir)
    print(dataset_dir, 'genearted')

img_idx=0
img_names=[]
img_labels=[]
for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        filepath = subdir + os.sep + file
        if filepath.endswith(".xml"):
            filename, file_extension = os.path.splitext(filepath)
            # print(filepath)
            try:
                tree = ET.parse(filepath)
                root = tree.getroot()
            except:

                print('xml error',filepath)
                continue

            for character in root.iter('character'):
                x=int(character.get('x'))
                y = int(character.get('y'))
                w = int(character.get('width'))
                h = int(character.get('height'))
                ch = character.get('char')
                if ch not in CLASSES:
                    continue
                try:
                    img = Image.open(filename+".jpg")
                except:
                    img = Image.open(filename + ".JPG")
                img.crop((x,y,x+w,y+h)).save(dataset_dir+'/img%d.png'%img_idx)
                img_names.append('img%d.png'%img_idx)
                img_idx+=1
                img_labels.append(ch)



with open( dataset_dir+'/'+dataset_dir+'.csv', 'w') as writeFile:
    writer = csv.writer(writeFile)
    writer.writerows(list(zip(img_names,img_labels)))
    print(dataset_dir+'/'+dataset_dir+'.csv','saved')