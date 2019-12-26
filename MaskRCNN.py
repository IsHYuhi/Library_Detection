import torch
import torchvision
import torchvision.models as models
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

#import ssl
#ssl._create_default_https_context = ssl._create_unverified_context
#セキュリティ上あまり良くないのでやりたくない
model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)#pretrained on COCO

#replace the classifier with a new one, that has num_classes which is user-defined
num_classes = 9
# get number of input features for thhe classifier
in_features = model.roi_heads.box_predictor.cls_score.in_features
#replace the pre-trained head with a new one
model.roi_heads.box_predictor = models.detection.faster_rcnn(in_features, num_classes)

#バックボーン追加

backbone = models.mobilenet_v2(pretrained=True).features#mobilenet_v2の最後の全結合層のinfeatures=1280
backbone.out_channels = 1280

#5 different sizes and 3 different aspect
anchor_generator = AnchorGenerator(size=((32, 64, 128, 256, 512),),
                                    aspect_ratios=((0.5, 1.0, 2.0),))

roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=[0],
                                                output_size = 7,
                                                sampling_ratio=2)

model = FasterRCNN(backbone,
                    num_classes=5,
                    rpn_anchor_generator=anchor_generator,
                    box_roi_pool=roi_pooler)

