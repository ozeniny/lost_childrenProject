import os
import random
import pickle

data_folder = r"D:\Graduation Project\missing2_withSieamese - Copy\outputs\all_faces"

all_files = os.listdir(data_folder)

person_to_images = {}
for filename in all_files:
    if filename.endswith(".jpg"):
        person_id = filename[:3]
        if person_id not in person_to_images:
            person_to_images[person_id] = []
        person_to_images[person_id].append(filename)
        
def generate_triplets(person_to_images, num_triplets=1000):
    triplets = []
    person_ids = list(person_to_images.keys())

    for _ in range(num_triplets):
        anchor_person = random.choice(person_ids)
        positive_person = anchor_person  
        negative_person = random.choice([p for p in person_ids if p != anchor_person])

        anchor_images = person_to_images[anchor_person]
        positive_images = person_to_images[positive_person]
        negative_images = person_to_images[negative_person]

        anchor = random.choice(anchor_images)
        positive = random.choice([img for img in positive_images if img != anchor])
        negative = random.choice(negative_images)

        triplets.append((anchor, positive, negative))

    return triplets

triplets = generate_triplets(person_to_images, num_triplets=5000)

for anchor, positive, negative in triplets[:5]:
    print(anchor, positive, negative)
    
output_file = r"D:\Graduation Project\missing2_withSieamese - Copy\triplet.pkl"

with open(output_file, 'wb') as file:
        pickle.dump(triplets, file)