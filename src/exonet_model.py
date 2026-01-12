import torch
from torch import nn

'''
Original code and model by Ansdell et al (2018)
https://gitlab.com/frontierdevelopmentlab/exoplanets

This code has been modified
'''
class ExtranetModel(nn.Module):
    '''
    PURPOSE: DEFINE EXTRANET MODEL ARCHITECTURE
    INPUT: GLOBAL + LOCAL LIGHT CURVES AND CENTROID CURVES, STELLAR PARAMETERS
    OUTPUT: BINARY CLASSIFIER
    '''    
    def __init__(self):

        ### initialize model
        super(ExtranetModel, self).__init__()

        ### define global convolutional lalyer
        self.fc_global = nn.Sequential(
            nn.Conv1d(2, 16, 5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv1d(16, 16, 5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(5, stride=2),
            nn.Conv1d(16, 32, 5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, 32, 5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(5, stride=2),
            nn.Conv1d(32, 64, 5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 64, 5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(5, stride=2),
            nn.Conv1d(64, 128, 5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv1d(128, 128, 5, stride=1, padding=2),
            nn.MaxPool1d(5, stride=2),
            nn.Conv1d(128, 256, 5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv1d(256, 256, 5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(5, stride=2),
        )

        ### define local convolutional lalyer
        self.fc_local = nn.Sequential(
            nn.Conv1d(2, 16, 5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv1d(16, 16, 5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(7, stride=2),
            nn.Conv1d(16, 32, 5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, 32, 5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(7, stride=2),
        )

        ### define fully connected layer that combines both views
        self.final_layer = nn.Sequential(
            nn.Linear(16582, 512), # 16586 --> 16582 since we do not use the proper motion data
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            ### need output of 1 because using BCE for loss
            nn.Linear(512, 1)
        ) # Sigmoid is moved outside to be consistent with the transformer code

    def forward(self, x:torch.tensor):
        x = x.unsqueeze(1).to(torch.float32)

        ### concatonate light curve and centroid data
        x_global = torch.cat([x[:,:,:2001],x[:,:,2001:4002]],dim=1)
        x_local = torch.cat([x[:,:,4002:4203],x[:,:,4203:4404]],dim=1)
        x_star = x_star = x[:,:,4404:]

        ### get outputs of global and local convolutional layers
        out_global = self.fc_global(x_global)
        out_local = self.fc_local(x_local)
        
        ### flattening outputs from convolutional layers into vector
        out_global = out_global.view(out_global.shape[0], -1)
        out_local = out_local.view(out_local.shape[0], -1)

        ### concatonate global and local views with stellar parameters
        out = torch.cat([out_global, out_local, x_star.squeeze(1)], dim=1)
        out = self.final_layer(out)

        return out
