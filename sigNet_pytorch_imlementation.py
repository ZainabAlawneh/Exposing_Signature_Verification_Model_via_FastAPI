import torch.nn as nn
import torch.nn.functional as F
import torch

class SigNet(nn.Module):
    def __init__(self):
        super(SigNet, self).__init__()
        self.conv1 = nn.Conv2d(1,96,11,1,0)
        self.LRN1 = nn.LocalResponseNorm(size=5,alpha=1e-4, beta=0.75, k=2)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv2 = nn.Conv2d(96,256,5,1,2) #Padding=2 keeps spatial size stable.
        self.LRN2 = nn.LocalResponseNorm(5,1e-4,0.75,2)
        self.pool2 = nn.MaxPool2d(kernel_size=3,stride=2)
        self.drop2 = nn.Dropout(p=0.3)

        self.conv3 = nn.Conv2d(256,384,3,1,1)

        self.conv4 = nn.Conv2d(384,256,3,1,1)

        self.pool3 = nn.MaxPool2d(kernel_size=3,stride=2)
        self.drop3 = nn.Dropout(p=0.3)
        
        
        with torch.no_grad():
            dummy = torch.zeros(1, 1, 189, 211) 
            out = F.relu(self.conv1(dummy))
            out = self.LRN1(out)
            out = self.pool1(out)

            out = F.relu(self.conv2(out))
            out = self.LRN2(out)
            out = self.pool2(out)
            out = self.drop2(out)

            out = F.relu(self.conv3(out))
            out = F.relu(self.conv4(out))
            out = self.pool3(out)
            out = self.drop3(out)

            self.flattened_size = out.numel()
            print("Flattened size:", self.flattened_size)

        """
        we must compute features size , (155,220) after all layers --> (256,17,25)
        so falttened size = 256 * 17 * 25 = 108800
        """

        self.FC1 = nn.Linear(self.flattened_size, 1024)
        self.dropFC = nn.Dropout(0.3)

        self.FC2 = nn.Linear(1024, 512)


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.LRN1(x)
        x = self.pool1(x)

        x = F.relu(self.conv2(x))
        x = self.LRN2(x)
        x = self.pool2(x)
        x = self.drop2(x)

        x = F.relu(self.conv3(x))

        x = F.relu(self.conv4(x))

        x = self.pool3(x)
        x = self.drop3(x)

        x = x.view(x.size(0), -1)

        x = F.relu(self.FC1(x))
        x = self.dropFC(x)

        x = self.FC2(x)

        x = F.normalize(x, p=2, dim=1)

        return x
    

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.embedding_net = SigNet()

    def forward(self, x1, x2):
      emb1 = self.embedding_net(x1)
      emb2 = self.embedding_net(x2)

      return emb1, emb2
    

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0, alpha=1.0, beta=1.0):
        super(ContrastiveLoss, self).__init__()  
        self.margin = margin
        self.alpha = alpha
        self.beta = beta

    def forward(self, emb1, emb2, y):
        distance = F.pairwise_distance(emb1, emb2)

        loss = self.alpha*(1-y)*torch.pow(distance,2) + self.beta*y *torch.pow(torch.clamp(self.margin - distance, min=0.0), 2)

        return loss.mean()
    

    

        

