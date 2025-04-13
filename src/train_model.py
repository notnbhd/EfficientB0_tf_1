import torch

from typing import Tuple


def train_model(model: torch.nn.Module,
                dataloader: torch.utils.data.DataLoader,
                criterion: torch.nn.Module, 
                optimizer: torch.optim.Optimizer,
                scheduler: torch.optim.lr_scheduler,
                device: torch.device) -> Tuple[float, float]:
    
    # Set the model to training mode
    model.train()

    train_loss, train_acc = 0, 0
    
    for batch, (X, y) in enumerate(dataloader):
        # Send data to target device
        X, y = X.to(device), y.to(device)

        #Forward pass
        y_pred = model(X)

        #calculate  and accumulate loss
        loss = criterion(y_pred, y)
        train_loss += loss.item() 

        #Optimizer zero grad
        optimizer.zero_grad()

        #Loss backward
        loss.backward()

        #Optimizer step
        optimizer.step()

        # Calculate and accumulate accuracy metric across all batches
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item()/len(y_pred)
            
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)

    # Scheduler step
    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        scheduler.step(train_loss)
    else:
        scheduler.step()

        
    return train_loss, train_acc