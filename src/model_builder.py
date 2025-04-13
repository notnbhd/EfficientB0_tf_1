import torch
import torchvision
import os

import data_setup, train


if __name__ == '__main__':
    data_setup.data_create()
    train_dataloader, test_dataloader, val_dataloader, class_names = data_setup.data_setup()

    weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
    model = torchvision.models.efficientnet_b0(weights=weights)


    # Freeze all base layers in the "features" section of the model (the feature extractor) by setting requires_grad=False
    for param in model.features.parameters():
        param.requires_grad = False

    # Unfreeze the last block:
    for param in model.features[6:].parameters():
        param.requires_grad = True


    output_shape = len(class_names)

    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(p=0.5, inplace=True),
        torch.nn.Linear(in_features=1280,
                        out_features=output_shape,
                        bias=True)
        )

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1)

    results, best_model = train.train(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        test_dataloader=test_dataloader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=25,
        device="cuda" if torch.cuda.is_available() else "cpu",
        patience=5  # Stop if no improvement for 5 consecutive epochs
    )

    torch.save(best_model.state_dict(), "best_model.pth")