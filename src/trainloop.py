import numpy as np
import torch
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score

def train_step(model,criterion,optimizer,dataloader,device):
    model.to(device)
    model.train()
    losses, accs = 0,0
    for batch,(X,y) in enumerate(dataloader):
        X,y = X.to(device),y.to(device)
        
        y_logit = model(X).squeeze()
        y_pred = torch.sigmoid(y_logit).round()

        loss = criterion(y_logit,y)
        losses += loss
        accs += accuracy_score(y.cpu(),y_pred.cpu().detach().numpy())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    losses /= len(dataloader)
    accs /= len(dataloader)
    return losses, accs

def val_step(model,criterion,dataloader,device):
    model.to(device)
    model.eval()
    losses, accs = 0,0
    with torch.inference_mode():
        for batch,(X,y) in enumerate(dataloader):
            X,y = X.to(device),y.to(device)
            y_logit = model(X).squeeze()
            y_pred = torch.sigmoid(y_logit).round()

            losses += criterion(y_logit,y)
            accs += accuracy_score(y.cpu(),y_pred.cpu().detach().numpy())
        losses /= len(dataloader)
        accs /= len(dataloader)
    return losses,accs

def train_loop(model,epochs,criterion,optimizer,device,train_dataloader,val_dataloader,scheduler=None):
    metrics = {'train_loss':[],
               'train_acc':[],
               'val_loss':[],
               'val_acc':[]}

    for epoch in tqdm(range(epochs)):
        print(f'Epoch {epoch}:')
        train_loss,train_acc = train_step(model,criterion,optimizer,train_dataloader,device)
        val_loss,val_acc = val_step(model,criterion,val_dataloader,device)

        if scheduler:
            scheduler.step(val_loss)
            
        metrics['train_loss'].append(train_loss.detach().item())
        metrics['train_acc'].append(train_acc)
        metrics['val_loss'].append(val_loss.detach().item())
        metrics['val_acc'].append(val_acc)
        print(f'Train Loss: {train_loss:.3f} | Train Acc: {train_acc:.2f} | Val Loss: {val_loss:.3f} | Val Acc: {val_acc:.2f}')
    return metrics