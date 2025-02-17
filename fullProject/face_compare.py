import shared_file as sf
from scipy.spatial.distance import cosine
import pickle


young_embeddings_file = r"D:\Graduation Project\missing2_withSieamese\outputs\young_embeddings.pkl"
old_embeddings_file = r"D:\Graduation Project\missing2_withSieamese\outputs\old_embeddings.pkl"


with open(young_embeddings_file, 'rb') as file:
    young_embeddings = pickle.load(file)

with open(old_embeddings_file, 'rb') as file:
    old_embeddings = pickle.load(file)

def compare_faces(embedding1, embedding2):
    return cosine(embedding1, embedding2)

top1_correct = 0
top5_correct = 0
total = len(young_embeddings)

for young_path, young_embedding in young_embeddings.items():
    similarities = []
    young_name = sf.os.path.splitext(sf.os.path.basename(young_path))[0]  
    
    for old_path, old_embedding in old_embeddings.items():
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