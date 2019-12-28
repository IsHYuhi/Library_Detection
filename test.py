import torchvision
import torch
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os
from models.MaskRCNN import get_model

CATEGORIES = {'1_overall':'red', '2_handwritten':'orange', '3_typography':'green', '4_illustration':'blue',
                '5_stamp':'yellow', '6_headline':'purple', '7_caption':'aqua', '8_textline':'gray', '9_table':'teal'}

BOOK_CATEGORIES = ['1_overall', '2_handwritten', '3_typography', '4_illustration',
                '5_stamp', '6_headline', '7_caption', '8_textline', '9_table']

if not os.path.isdir("./predicted/"):
            os.mkdir("./predicted")

imgs = list(sorted(os.listdir(os.path.join("./", "train_images"))))
num_classes = 10
device = torch.device('cpu')
model = get_model(num_classes)
PATH = "./checkpoint/model_9.pth"
model.load_state_dict(torch.load(PATH, map_location='cpu'))
model.to(device)
model.eval()
model = model.to(device)

for img_name in imgs:
    img_path = os.path.join("./", "train_images", img_name)
    img = Image.open(img_path).convert("RGB")
    image_tensor = torchvision.transforms.functional.to_tensor(img)
    x = [image_tensor.to(device)]

    prediction = model(x)[0]
    print(prediction)

    bboxes_np = prediction['boxes'].to(torch.int16).cpu().numpy()
    labels_np = prediction['labels'].byte().cpu().numpy()
    scores_np = prediction['scores'].cpu().detach().numpy()
    bboxes = []
    labels = []
    scores = []

    draw = ImageDraw.Draw(img)
    for i, bbox in enumerate(bboxes_np):
        score = scores_np[i]
        if score < 0.8:
            continue

        label = labels_np[i]
        bboxes.append([bbox[1], bbox[0], bbox[3], bbox[2]])
        draw.rectangle([(bbox[0], bbox[1]), (bbox[2], bbox[3])], outline=CATEGORIES[BOOK_CATEGORIES[label-1]],  width=5)
        labels.append(label - 1)
        draw.font = ImageFont.truetype('./font/abel-regular.ttf', 30)
        draw.text((bbox[0], bbox[1]), BOOK_CATEGORIES[label-1]+":"+ str(score), fill='black')
        scores.append(score)

    bboxes = np.array(bboxes)
    labels = np.array(labels)
    scores = np.array(scores)
    #img.show()

    img.save('./predicted/'+ img_name)