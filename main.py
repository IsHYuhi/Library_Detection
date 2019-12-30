from utils.engine import train_one_epoch, evaluate
from utils import utils
from utils import transforms as T
from utils.dataset import BookDataset
from models.MaskRCNN import get_model
import torch
import os
def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())#一度全てToTensor()
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))#0.5の確率でRandomHorizontalFlip
        transforms.append(T.RandomErasing())#default.p=0.5
        #transforms.append(T.ColorJitter(brightness=0.125, contrast=0.125, saturation=.05, hue=.05))
    return T.Compose(transforms)

def main():
    #GPU or CPU
    #device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    device = torch.device('cpu')
    num_classes = 10

    dataset = BookDataset('./', get_transform(train=True))
    dataset_test = BookDataset('./', get_transform(train=False))

    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:-50])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn
    )

    model = get_model(num_classes)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    num_epochs = 10
    if not os.path.isdir("./checkpoint/"):
            os.mkdir("./checkpoint")
    for epoch in range(num_epochs):
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)

        lr_scheduler.step()

        evaluate(model, data_loader_test, device=device)

        torch.save(model.state_dict(), "./checkpoint/model_"+ str(epoch) +".pth")
    print("Done!")

if __name__ == '__main__':
    main()