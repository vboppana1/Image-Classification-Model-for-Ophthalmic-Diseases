import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

csv_file_path = 'full_df.csv'
df = pd.read_csv(csv_file_path)
plt.xlim(0, 500)
plt.ylim(0, 300)

while True:
    answer = int(input("What patient would you like to see? "))
    if answer in df['ID'].values:
        break
    else:
        print(f"Patient ID {answer} not found. Please enter another ID.")

img_left = mpimg.imread('Training Images/' + str(answer) + '_left.jpg')
img_right = mpimg.imread('Training Images/' + str(answer) + '_right.jpg')

patient_rows = df[df['ID'] == answer]
patient_age = patient_rows['Patient Age'].iloc[0]
patient_sex = patient_rows['Patient Sex'].iloc[0]

patient_conditions = {}

for i in [0, 1]:
    patient_diagnoses = patient_rows['target'].iloc[i][1:-1].split(", ")
    eye = 'Left' if 'left' in patient_rows['filename'].iloc[i] else 'Right'
    possible_conditions = ['Normal','Diabetes','Glaucoma','Cataract','Age related Macular Degeneration','Hypertension','Pathological Myopia','Other']
    patient_conditions_eye = []
    for i_1,i_2 in zip(patient_diagnoses, possible_conditions):
        if int(i_1):
            patient_conditions_eye.append(i_2)
    patient_conditions[eye] = patient_conditions_eye


plt.imshow(img_left, extent=[25, 235, 150, 300])     
plt.imshow(img_right, extent=[265, 475, 150, 300])    
plt.axis('off')

plt.figtext(0.6, 0.44, f'Patient ID: {answer}')
plt.figtext(0.15, 0.44, f'Age: {patient_age}')
plt.figtext(0.15, 0.30, f'Left Eye Conditions: {patient_conditions['Left']}')
plt.figtext(0.6, 0.30, f'Right Eye Conditions: {patient_conditions['Right']}')


plt.show()


# from PIL import Image
# import os
# from collections import defaultdict

# # Set this to your folder containing the images
# image_folder = "Training Images"

# # Track min/max dimensions and frequency
# min_width = float('inf')
# min_height = float('inf')
# max_width = 0
# max_height = 0
# size_counts = defaultdict(int)

# for filename in os.listdir(image_folder):
#     if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
#         try:
#             filepath = os.path.join(image_folder, filename)
#             with Image.open(filepath) as img:
#                 width, height = img.size

#                 # Update min/max
#                 if width < min_width:
#                     min_width = width
#                 if height < min_height:
#                     min_height = height
#                 if width > max_width:
#                     max_width = width
#                 if height > max_height:
#                     max_height = height

#                 # Count unique size
#                 size_counts[(width, height)] += 1

#         except Exception as e:
#             print(f"Error with {filename}: {e}")

# print(f"\n🟢 Smallest dimension: {min_width}x{min_height}")
# print(f"🔴 Largest dimension: {max_width}x{max_height}")

# # Optional: print how many times each size appears
# print("\n📊 Most common sizes (top 10):")
# sorted_sizes = sorted(size_counts.items(), key=lambda x: x[1], reverse=True)
# for i, ((w, h), count) in enumerate(sorted_sizes[:10]):
#     print(f"  {w}x{h} — {count} images")

# print(f"\n📦 Total unique dimension types: {len(size_counts)}")

