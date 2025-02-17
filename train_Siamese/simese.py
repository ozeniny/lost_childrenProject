import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from facenet_pytorch import InceptionResnetV1
import pickle
import torch.optim as optim
import os


class SiameseNetwork(nn.Module):
    def __init__(self, embedding_dim=512):
        super(SiameseNetwork, self).__init__()
        self.fc = nn.Linear(embedding_dim, 256)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(256) 
        self.out = nn.Linear(256, 128) 

    def forward(self, anchor, positive, negative):
        anchor = self._embed(anchor)
        positive = self._embed(positive)
        negative = self._embed(negative)
        return anchor, positive, negative
    
    def _embed(self, x):
        x = self.relu(self.fc(x))
        x = self.bn(x)
        x = self.out(x)
        return F.normalize(x, p=2, dim=1)

    
    
 
class TripletDataset(Dataset):
    def __init__(self, triplets, embeddings_dict):
        self.triplets = triplets
        self.embeddings_dict = embeddings_dict

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        anchor_filename, positive_filename, negative_filename = self.triplets[idx]
        
        anchor_path = os.path.join(r"D:\Graduation Project\missing2_withSieamese - Copy\outputs\all_faces", anchor_filename)
        positive_path = os.path.join(r"D:\Graduation Project\missing2_withSieamese - Copy\outputs\all_faces", positive_filename)
        negative_path = os.path.join(r"D:\Graduation Project\missing2_withSieamese - Copy\outputs\all_faces", negative_filename)
        
        anchor = torch.tensor(self.embeddings_dict[anchor_path], dtype=torch.float32)
        positive = torch.tensor(self.embeddings_dict[positive_path], dtype=torch.float32)
        negative = torch.tensor(self.embeddings_dict[negative_path], dtype=torch.float32)
        return anchor, positive, negative
    
with open(r"D:\Graduation Project\missing2_withSieamese - Copy\triplet.pkl", "rb") as file:
    triplets = pickle.load(file)

with open(r"D:\Graduation Project\missing2_withSieamese - Copy\outputs\all_faces.pkl", "rb") as file:
    embeddings_dict = pickle.load(file)


triplet_dataset = TripletDataset(triplets, embeddings_dict)
train_loader = DataLoader(triplet_dataset, batch_size=8, shuffle=True)

siamese_net = SiameseNetwork()
triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)

optimizer = optim.Adam(siamese_net.parameters(), lr=0.001)

num_epochs = 20
for epoch in range(num_epochs):
    siamese_net.train()
    running_loss = 0.0
    for anchor, positive, negative in train_loader:
        optimizer.zero_grad()
        anchor_embed, positive_embed, negative_embed = siamese_net(anchor, positive, negative)
        loss = triplet_loss(anchor_embed, positive_embed, negative_embed)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")
    
torch.save(siamese_net.state_dict(), "siamese_net.pth")
