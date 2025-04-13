import torch
import torchvision
import torchvision.transforms as transforms


def image_classify(img):
    model = torchvision.models.efficientnet_b0(weights='DEFAULT')
    class_names = ['pho', 'ramen', 'spaghetti_carbonara']
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(p=0.2, inplace=True),
        torch.nn.Linear(in_features=1280, out_features=len(class_names), bias=True)
        )
    model.load_state_dict(torch.load('./src/best_model.pth'))

    image_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
        ])

    img = image_transform(img)
    img = img.unsqueeze(0)

    model.eval()
    with torch.no_grad():
        output = model(img)
        _, pred = torch.max(output, 1)
    
    return class_names[pred.item()]