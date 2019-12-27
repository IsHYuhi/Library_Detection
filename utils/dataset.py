import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image, ImageDraw
import json
count = 0

#refered to List of common HTML color codes
CATEGORIES = {'1_overall':'red', '2_handwritten':'blue', '3_typography':'green', '4_illustration':'yellow',
                '5_stamp':'orange', '6_headline':'purple', '7_caption':'aqua', '8_textline':'lime', '9_table':'teal'}

BOOK_CATEGORIES = ['1_overall', '2_handwritten', '3_typography', '4_illustration',
                '5_stamp', '6_headline', '7_caption', '8_textline', '9_table']

class BookDataset(object):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        #画像をロードし、ラベル付きのものと揃えてpathをソートする
        self.imgs = list(sorted(os.listdir(os.path.join(root, "train_images"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "train_annotations"))))

    def __getitem__(self, idx):
        #load images an masks
        #idxの画像のpathをimg_pathに入れる
        img_path = os.path.join(self.root, "train_images", self.imgs[idx])
        mask_path = os.path.join(self.root, "train_annotations", self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        #draw = ImageDraw.Draw(img)

        #mask = image.open(mask_path)
        #mask = np.array(mask)
        #インスタンスをユニークなカラーにエンコードする
        #obj_ids = np.unique(mask)
        #obj_ids = obj_ids[1:]
        #binary masksのセットに分ける
        #masks = mask == obj_ids[:, None, None]
        #get bounding box coordinates for each mask
        #num_objs = len(obj_ids)

        boxes = []
        #categories = []
        labels = []
        json_file = open(mask_path, 'r')
        json_objects = json.load(json_file)
        for json_object in json_objects['labels']:
            category = json_object['category']
            x1 = json_object['box2d']['x1']
            x2 = json_object['box2d']['x2']
            y1 = json_object['box2d']['y1']
            y2 = json_object['box2d']['y2']

            #draw.rectangle([(x1, y1), (x2, y2)], outline=CATEGORIES[category],  width=5) #領域を描画確認
            #print(category)
            #print([x1, y1, x2, y2])

            boxes.append([x1, y1, x2, y2])
            labels.append(BOOK_CATEGORIES.index(category)+1)
            # if not category in categories:
            #     categories.append(category)

        #img.show()
        #print(categories)
        #print(len(categories))

        #num_objs = len(categories) #インスタンスで別のオブジェクトとして捉えているので(？)
        num_objs = len(labels)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        #only one class if you wanna clasify multiple label, you can allocate label number to each object
        #labels = torch.ones((num_objs,), dtype=torch.int64) #今回は複数ラベル
        #print(labels)
        labels = torch.tensor(labels, dtype=torch.int64)
        #print(labels)
        #(option)masks

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1])*(boxes[:, 2]-boxes[:,0])

        #all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
        #print(iscrowd)

        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['image_id'] = image_id
        target['area'] = area
        target['iscrowd'] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)


#if __name__ == '__main__':
    #test = BookDataset('./', None)
    #test.__getitem__(0)
    #for i in range(5):
        #test.__getitem__(i)