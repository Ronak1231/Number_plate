import os
import xml.etree.ElementTree as ET
from PIL import Image
import shutil
from tqdm import tqdm

def convert(size, box):
    dw, dh = 1. / size[0], 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    return x * dw, y * dh, w * dw, h * dh

def convert_voc_to_yolo(xml_dir, img_dir, out_img_dir, out_lbl_dir):
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_lbl_dir, exist_ok=True)
    
    for file in tqdm(os.listdir(xml_dir)):
        if not file.endswith(".xml"):
            continue

        xml_path = os.path.join(xml_dir, file)
        tree = ET.parse(xml_path)
        root = tree.getroot()
        filename = root.find("filename").text

        image_path = os.path.join(img_dir, filename)
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            continue

        img = Image.open(image_path)
        w, h = img.size

        yolo_txt = []
        for obj in root.iter('object'):
            cls = obj.find('name').text
            if cls.lower() != "number_plate":
                continue
            xmlbox = obj.find('bndbox')
            try:
                b = (
                    float(xmlbox.find('xmin').text),
                    float(xmlbox.find('xmax').text),
                    float(xmlbox.find('ymin').text),
                    float(xmlbox.find('ymax').text)
                )
                yolo_bbox = convert((w, h), b)
                yolo_txt.append(f"0 {' '.join(f'{a:.6f}' for a in yolo_bbox)}")
            except Exception as e:
                print(f"Error in {file}: {e}")
                continue

        # Copy image
        shutil.copy(image_path, os.path.join(out_img_dir, filename))

        # Save label
        base_filename = os.path.splitext(filename)[0]
        label_path = os.path.join(out_lbl_dir, base_filename + '.txt')
        with open(label_path, 'w') as f:
            f.write('\n'.join(yolo_txt))

# Run conversion
convert_voc_to_yolo(
    xml_dir='Annotations/Annotations',
    img_dir='Indian_Number_Plates/Sample_Images',
    out_img_dir='indian_number_plates/images/train',
    out_lbl_dir='indian_number_plates/labels/train'
)
