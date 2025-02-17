import cv2
import os
from mtcnn import MTCNN
from keras_facenet import FaceNet
from facenet_pytorch import InceptionResnetV1
from insightface.app import FaceAnalysis
import numpy as np
from skimage import exposure
from torchvision import transforms
import torch

detector = MTCNN()

#embedder = FaceNet()

embedder2 = InceptionResnetV1(pretrained='vggface2').eval()

#app = FaceAnalysis(name='buffalo_l')
#app.prepare(ctx_id=0, det_size=(320, 320))


transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((160, 160)), 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  
])


def extract_face(img, box):
    x, y, width, height = box
    face_region = img[y:y+height, x:x+width]    
    
    face_region = cv2.resize(face_region, (160, 160))
    face_region = face_region.astype('float32')
    
    return face_region

def normalize_embedding(embedding):

    max_abs = np.max(np.abs(embedding))
    if max_abs == 0:
        return embedding
    return embedding / max_abs


def get_embedding_InceptionResnetV1(face_region):

    face_tensor = transform(face_region).unsqueeze(0) 
    
    with torch.no_grad():  
        embedding = embedder2(face_tensor).squeeze().numpy()  
    
    return embedding

def get_embedding_InsightFace (face_region):
    
    faces = app.get(face_region)
    if faces:
        return faces[0].embedding
    else:
        print("No face detected by InsightFace.")
        return None
