import torch
from torch import dtype
import torch.nn as nn

def train(data_loader, model, optimizer, device):
    """
    :param data_loader: torch dataloader
    :param model: model(lstm model)
    :param optimizer: torch optimizer, i.e. SGD, ADAM etc.
    
    """
    # set-up the model to training mode
    model.train()
    
    # go through each batch of data in data_loader
    for data in data_loader:
        reviews = data["review"]
        targets = data["target"]
        
        reviews = reviews.to(device, dtype=torch.long)
        targets = targets.to(device, dtype=torch.float)
        
        # .zero_grad to empty the grad_fn after each epoch
        optimizer.zero_grad()
        
        preds = model(reviews)
        
        loss = nn.BCELoss()(
            preds,
            targets.view(-1, 1)
        )   
        
        loss.backward()     
        
        optimizer.step()
        
def evaluate(data_loader, model, device):
    
    final_predictions = []
    final_targets = []
    
    model.eval()
    
    with torch.no_grad():
        for data in data_loader:
            reviews = data["review"]
            targets = data["target"]
            
            reviews = reviews.to(device, dtype=torch.long)
            targets = targets.to(device, dtype=torch.float)
            
            predictions = model(reviews)
            
            predictions = predictions.cpu().numpy().tolist()
            targets = data["targets"].cpu().numpy.tolist()
            
            final_predictions.extend(predictions)
            final_targets.extend(targets)
            
    return final_predictions, final_targets


            
            
            
        
        
        