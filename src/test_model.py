import torch

from typing import Tuple


def test_model(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               criterion: torch.nn.Module,
               device: torch.device) -> Tuple[float, float]:
    #eval mode on
    model.eval()
    test_loss, test_acc = 0, 0

    with torch.inference_mode():
        #loop through DataLoader batches:
        for batch, (X, y) in enumerate(dataloader):
            # Send data to target device
            X, y = X.to(device), y.to(device)

            # Forward pass
            test_pred = model(X)

            # Calculate and accumulate loss
            loss = criterion(test_pred, y)
            test_loss += loss.item()

            # Calculate and accumulate accuracy metric across all batches
            y_pred_labels = test_pred.argmax(dim=1)
            test_acc += (y_pred_labels == y).sum().item()/len(y_pred_labels)
    
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    return test_loss, test_acc