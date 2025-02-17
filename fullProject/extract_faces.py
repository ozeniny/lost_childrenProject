import shared_file as sf
import os
import pickle

def process_image_faces(image_folder, output_folder, output_file):
    os.makedirs(output_folder, exist_ok=True)

    embeddings_dict = {}

    image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png'))]

    for image_path in image_paths:
        img = sf.cv2.imread(image_path)
        if img is None:
            print(f"Failed to load image: {image_path}")
            continue

        faces = sf.detector.detect_faces(img)


        if len(faces) == 0:
            print(f"No faces detected in image: {image_path}")
            continue

        
        face = faces[0] 
        input_face_region = sf.extract_face(img, face['box'])
        
        # InsightFace
        #input_face_embedding = sf.get_embedding_InsightFace(input_face_region)
        
        # InceptionResnetV1
        input_face_embedding = sf.get_embedding_InceptionResnetV1(input_face_region)
        
        # FaceNet
        #input_face_embedding = sf.embedder.embeddings(sf.np.expand_dims(input_face_region, axis=0))[0]

        if input_face_embedding is None:
            print(f"No face embedding found for image: {image_path}")
            continue
        
        input_face_embedding = sf.normalize_embedding(input_face_embedding)

       
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        file_name = f"{base_name}.jpg"
        face_output_path = os.path.join(output_folder, file_name)

        sf.cv2.imwrite(face_output_path, sf.cv2.cvtColor(input_face_region, sf.cv2.COLOR_RGB2BGR))
        print(f"Saved face image: {face_output_path}")

        embeddings_dict[face_output_path] = input_face_embedding

    with open(output_file, 'wb') as file:
        pickle.dump(embeddings_dict, file)
    print(f"Saved {len(embeddings_dict)} face embeddings to {output_file}")

young_folder = r"D:\Graduation Project\lost_children_dataset\lost_children\Young"
old_folder = r"D:\Graduation Project\lost_children_dataset\lost_children\Old"
young_output_folder = r"D:\Graduation Project\missing2_withSieamese\outputs\young_faces"
old_output_folder = r"D:\Graduation Project\missing2_withSieamese\outputs\old_faces"
young_embeddings_file = r"D:\Graduation Project\missing2_withSieamese\outputs\young_embeddings.pkl"
old_embeddings_file = r"D:\Graduation Project\missing2_withSieamese\outputs\old_embeddings.pkl"


process_image_faces(young_folder, young_output_folder, young_embeddings_file)
process_image_faces(old_folder, old_output_folder, old_embeddings_file)