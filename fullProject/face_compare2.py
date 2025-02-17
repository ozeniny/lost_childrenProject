import shared_file as sf
from scipy.spatial.distance import cosine
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F

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

siamese_net = SiameseNetwork()
siamese_net.load_state_dict(torch.load("siamese_net.pth"))
siamese_net.eval()

young_embeddings_file = r"D:\Graduation Project\missing2_withSieamese\outputs\young_embeddings.pkl"
old_embeddings_file = r"D:\Graduation Project\missing2_withSieamese\outputs\old_embeddings.pkl"

with open(young_embeddings_file, 'rb') as file:
    young_embeddings = pickle.load(file)

with open(old_embeddings_file, 'rb') as file:
    old_embeddings = pickle.load(file)

def compute_embedding(embedding):
    with torch.no_grad():
        embedding_tensor = torch.tensor(embedding, dtype=torch.float32).unsqueeze(0)
        return siamese_net._embed(embedding_tensor).squeeze().numpy()

new_young_embeddings = {path: compute_embedding(embedding) for path, embedding in young_embeddings.items()}
new_old_embeddings = {path: compute_embedding(embedding) for path, embedding in old_embeddings.items()}

def compare_faces(embedding1, embedding2):
    return cosine(embedding1, embedding2)

top1_correct = 0
top5_correct = 0
total = len(new_young_embeddings)

for young_path, young_embedding in new_young_embeddings.items():
    similarities = []
    young_name = sf.os.path.splitext(sf.os.path.basename(young_path))[0]

    for old_path, old_embedding in new_old_embeddings.items():
        old_name = sf.os.path.splitext(sf.os.path.basename(old_path))[0]
        dist = compare_faces(young_embedding, old_embedding)
        similarities.append((dist, old_path))
    similarities.sort(key=lambda x: x[0])

    if similarities and sf.os.path.splitext(sf.os.path.basename(similarities[0][1]))[0] == young_name:
        top1_correct += 1

    top5_matches = [sf.os.path.splitext(sf.os.path.basename(path))[0] for dist, path in similarities[:5]]
    if young_name in top5_matches:
        top5_correct += 1

top1_accuracy = top1_correct / total
top5_accuracy = top5_correct / total

print(f"Top-1 Accuracy: {top1_accuracy:.4f}")
print(f"Top-5 Accuracy: {top5_accuracy:.4f}")