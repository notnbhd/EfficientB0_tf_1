import torch
import copy

from tqdm.auto import tqdm
from typing import Dict, List


from train_model import train_model
from test_model import test_model



def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          val_dataloader: torch.utils.data.DataLoader,  # Changed from test_dataloader
          test_dataloader: torch.utils.data.DataLoader,  # Added separate test dataloader
          criterion: torch.nn.Module,
          optimizer: torch.optim.Optimizer,
          scheduler: torch.optim.lr_scheduler,
          epochs: int,
          device: torch.device,
          patience: int = 5) -> Dict[str, List[float]]:
    
    # Create a dictionary to store the results
    results = {"train_loss": [], 
               "train_acc": [],
               "val_loss": [],   
               "val_acc": [],    
               "test_loss": [],
               "test_acc": []    
              }
    
    # Early stopping variables
    best_val_acc = 0.0  # Changed from best_test_acc
    best_model_wts = copy.deepcopy(model.state_dict())
    counter = 0
    
    # Loop through the epochs
    for epoch in tqdm(range(epochs)):
        # Train the model
        train_loss, train_acc = train_model(model=model,
                                           dataloader=train_dataloader,
                                           criterion=criterion,
                                           optimizer=optimizer,
                                           scheduler=scheduler,
                                           device=device)
        
        # Validate the model
        val_loss, val_acc = test_model(model=model,  # Using test_model function for validation
                                       dataloader=val_dataloader,
                                       criterion=criterion,
                                       device=device)

        # Print the results
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train loss: {train_loss:.4f}, Train accuracy: {train_acc:.4f}")
        print(f"Val loss: {val_loss:.4f}, Val accuracy: {val_acc:.4f}")

        # Append the results to the dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["val_loss"].append(val_loss)
        results["val_acc"].append(val_acc)
        
        # Early stopping check - using validation accuracy
        if val_acc > best_val_acc:
            print(f"Validation accuracy improved from {best_val_acc:.4f} to {val_acc:.4f}. Saving model...")
            best_val_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            counter = 0
        else:
            counter += 1
            print(f"Validation accuracy did not improve. Counter: {counter}/{patience}")
            
            if counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break
    
    # Load the best model weights
    print(f"Loading best model with validation accuracy: {best_val_acc:.4f}")
    model.load_state_dict(best_model_wts)
    
    # Evaluate on test set
    test_loss, test_acc = test_model(model=model,
                                    dataloader=test_dataloader,
                                    criterion=criterion,
                                    device=device)
    
    results["test_loss"].append(test_loss)
    results["test_acc"].append(test_acc)
    print(f"Test loss: {test_loss:.4f}, Test accuracy: {test_acc:.4f}")
    
    return results, model