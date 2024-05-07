import torch
from torchvision.transforms import v2
import pandas as pd
import os
from PIL import Image


new_rows= []

projections= pd.read_csv("./indiana_projections.csv")

for filename in os.listdir('./augmen_images'):
    original_filename, index_with_extension = filename.rsplit('-', 1)
    original_filename = original_filename + '.dcm'+ '.png'
    if original_filename in projections['filename'].values:

        row = projections[projections['filename'] == original_filename].iloc[0]
        filename= filename.replace(".png", ".dcm.png")
        new_rows.append({
            'uid': row['uid'],
            'filename': filename,
            'projection': row['projection']
        })

        
augmented_projections = pd.DataFrame(new_rows)
augmented_projections.to_csv('./augmented_projections.csv', index=False)

print("hogaya")

clean_reports= pd.read_csv('./clean_indiana_reports.csv')

df_new=pd.merge(clean_reports,projections, on='uid')
df_augm=pd.merge(df_new,augmented_projections, on='uid')
df_total = pd.concat([df_new, df_augm], ignore_index=True)
df_total.to_csv('./df_total.csv')
print("hogaya")

df_total_2 = df_total.groupby('Problems')['filename'].apply(list)
df_classes= df_new.groupby('Problems')['filename'].apply(list)

rotation_1= [5,10,15,20,25,27,30,35,38,40,45,50]
rotation_2= [5,10,15,20,25]
rotation_4= [10,20,30]
contrast_2= [0.5,0.6,0.7,0.8,0.9]
contrast_3= [0.5,0.6,0.7]
# saturation= [0.5,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1]
# saturation_2= [0.4,0.5,0.6,0.7,0.8]
# saturation_3= [0.4,0.6,0.8]
bright_2= [0.4,0.5,0.6,0.7,0.8]
bright_3= [0.4,0.5,0.6]
bright_4=[0.5,0.6,0.7,0.8]
# hue= [0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8,2.0]
# for column in df_total.columns:
#     print(column)

# filename_to_check = '2570_IM-1073-1001.dcm.png'

# # Filter the DataFrame based on the filename
# filtered_df = df_total[df_total['filename'] == filename_to_check]

# # Check if the filename exists in the DataFrame
# if not filtered_df.empty:
#     # Get the 'Problem' value for the filename
#     file_problem = filtered_df.iloc[0]['Problems']
#     print(f"The Problem for filename '{filename_to_check}' is '{file_problem}'.")
# else:
#     print(f"Filename '{filename_to_check}' not found in the DataFrame.")

# x= df_total['Problems'].value_counts()
# print(x['Cardiac Shadow;Pericardial Effusion;Pulmonary Congestion;Markings;Aorta, Thoracic;Cardiomegaly;Pulmonary Edema;Thickening'])

# filtered_df_total_2 = {problem: filenames for problem, filenames in df_total_2.items() if len(filenames) < 10}
for problem, filenames in df_total_2.items():
    num_images = len(filenames)
    # print('filenames',filenames)
    print('inside loop')
    while num_images<10:
        for filename in filenames:
            k=100
            
            filename= filename.replace(".dcm", "")
            img_path = os.path.join('./images/images_normalized', filename)
            if os.path.exists(img_path):
                img_path = img_path
            else:
                img_path = os.path.join('./augmen_images', filename)

            img = Image.open(img_path)
            filename= filename.replace(".png","")

            if num_images== 1 :
                for i in rotation_1:
                    transforms= v2.Compose([
                        v2.RandomRotation(i),
                        v2.ToImage()
                    ])
                    print('transform begins')
                    new_img= transforms(img)
                    print('transform ends')
                    to_pil= v2.ToPILImage()
                    print('converted to pil')
                    new_img= to_pil(new_img)
                    print('yay going to save')
                    new_img.save(os.path.join('./aug_images',f'{filename}-{problem}-{k}.png'))
                    print('saved hurray!')
                    # new_img.show() 
                    k=k+1 
                    print(k) 

            if num_images == 2 :
                for i in bright_2:
                    transforms= v2.Compose([
                        v2.ColorJitter(brightness=i),
                        v2.ToTensor()
                    ])
                    print('transform begins')
                    new_img= transforms(img)
                    print('transform ends')
                    to_pil= v2.ToPILImage()
                    print('converted to pil')
                    new_img= to_pil(new_img)
                    print('yay going to save')
                    new_img.save(os.path.join('./aug_images',f'{filename}-{problem}-{k}.png'))
                    print('saved hurray!')
                    # new_img.show()
                    k=k+1
                    print(k)
            
            if num_images == 3:
                for i in bright_4:
                    transforms= v2.Compose([
                        v2.ColorJitter(brightness=i),
                        v2.ToTensor()
                    ])
                    print('transform begins')
                    new_img= transforms(img)
                    print('transform ends')
                    to_pil= v2.ToPILImage()
                    print('converted to pil')
                    new_img= to_pil(new_img)
                    print('yay going to save')
                    new_img.save(os.path.join('./aug_images',f'{filename}-{problem}-{k}.png'))
                    print('saved hurray!')
                    # new_img.show()
                    k=k+1
                    print(k)

            if num_images == 4:
                for i in contrast_3:
                    transforms= v2.Compose([
                        v2.ColorJitter(contrast=i),
                        v2.ToTensor()
                    ])
                    print('transform begins')
                    new_img= transforms(img)
                    print('transform ends')
                    to_pil= v2.ToPILImage()
                    print('converted to pil')
                    new_img= to_pil(new_img)
                    print('yay going to save')
                    new_img.save(os.path.join('./aug_images',f'{filename}-{problem}-{k}.png'))
                    print('saved hurray!')
                    # new_img.show()
                    k=k+1
                    print(k)

            if 5<= num_images <=8:
                for i in rotation_2:
                    transforms= v2.Compose([
                        v2.ColorJitter(contrast=i),
                        v2.ToTensor()
                    ])
                    print('transform begins')
                    new_img= transforms(img)
                    print('transform ends')
                    to_pil= v2.ToPILImage()
                    print('converted to pil')
                    new_img= to_pil(new_img)
                    print('yay going to save')
                    new_img.save(os.path.join('./aug_images',f'{filename}-{problem}-{k}.png'))
                    print('saved hurray!')
                    # new_img.show()
                    k=k+1
                    print(k)