''' IMPORT LIBRARIES '''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix,accuracy_score,recall_score,precision_score
import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader
import os
from tqdm.notebook import tqdm

from dataset import LightCurveDataset
from exonet_model import ExtranetModel
from trainloop import train_loop

os.makedirs('./figs',exist_ok=True)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

''' DEFINE HYPERPARAMETER '''
BATCH_SIZE = 32
LR = 1e-5
EPOCHS = 100
FILENAME = 'exonet'

''' DATA READING '''
train_frame = pd.read_csv('../data/train_dataset_full.csv')
val_frame = pd.read_csv('../data/val_dataset_full.csv')
test_frame = pd.read_csv('../data/test_dataset_full.csv')

X_train = train_frame.drop('label',axis=1)
y_train = train_frame['label']

X_val = val_frame.drop('label',axis=1)
y_val = val_frame['label']

X_test = test_frame.drop('label',axis=1)
y_test = test_frame['label']

''' LOAD DATA INTO DATALOADER '''
dataset_train = LightCurveDataset(X_train,y_train)
dataset_val = LightCurveDataset(X_val,y_val)
dataset_test = LightCurveDataset(X_test,y_test)

loader_train = DataLoader(dataset=dataset_train,
                         batch_size=BATCH_SIZE,
                         shuffle=True,
                         drop_last=True)
loader_val = DataLoader(dataset=dataset_val,
                        batch_size=BATCH_SIZE,
                        shuffle=False,
                        drop_last=True)
loader_test = DataLoader(dataset=dataset_test,
                         batch_size=BATCH_SIZE,
                         shuffle=False,
                         drop_last=False)

''' DEFINE MODEL'''
MODEL = ExtranetModel().to(device)
CRITERION = nn.BCEWithLogitsLoss()
OPTIMIZER = torch.optim.Adam(params=MODEL.parameters(),lr=LR)
SCHEDULER = torch.optim.lr_scheduler.ReduceLROnPlateau(OPTIMIZER,mode='min',factor=0.1,min_lr=1e-7,patience=15)

''' TRAIN '''
metrics = train_loop(MODEL,EPOCHS,CRITERION,OPTIMIZER,device,loader_train,loader_val,scheduler=SCHEDULER)

plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(metrics['train_loss'],label='Train',color='C0')
plt.plot(metrics['val_loss'],label='Val',color='C1')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(loc='upper right')

plt.subplot(1,2,2)
plt.plot(metrics['train_acc'],label='Train',color='C0')
plt.plot(metrics['val_acc'],label='Val',color='C1')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')

plt.suptitle('Progress Plot')
plt.tight_layout()
plt.savefig(f'./figs/{FILENAME}_losscurve.png')
plt.close()

''' EVALUATE '''
MODEL.eval()
accs = []
preds = []
trues = []
with torch.inference_mode():
    for X,y in loader_test:
        X,y = X.to(device),y.to(device)
        y_logit = MODEL(X).squeeze()
        y_preds = torch.sigmoid(y_logit).round()
        
        accs.append(accuracy_score(y.cpu(),y_preds.cpu().detach().numpy()))
        trues.append(y.cpu())
        preds.append(y_preds.cpu().detach().numpy())
preds = np.hstack([*preds])
trues = np.hstack([*trues])

avg_accs = np.mean(accs)
all_accs = accuracy_score(trues,preds)
all_recalls = recall_score(trues,preds)
all_precision = precision_score(trues,preds)

confmat = confusion_matrix(trues,preds)
classes = ['False Positive','Planet']

plt.figure(figsize=(7,6))
sns.heatmap(confmat,annot=True,cmap='Blues')
plt.xticks([0.5,1.5],classes)
plt.yticks([0.5,1.5],classes)
plt.title(f'Accuracy: {all_accs:.4f}\nRecall: {all_recalls:.4f}\nPrecision: {all_precision:.4f}')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.tight_layout()
plt.savefig(f'./figs/{FILENAME}_confmat.png')
plt.close()

''' SAVE '''
os.makedirs('saves',exist_ok=True)
torch.save(MODEL.state_dict(), f'saves/vit_latefusion_{FILENAME}.pth')
