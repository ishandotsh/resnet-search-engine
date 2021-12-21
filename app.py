import streamlit as st
import torch
from torchvision import datasets, transforms
from torchvision.models import resnet34
import numpy as np
from PIL import Image
 
import random

st.set_page_config(
     page_title="ResNet Image Search Engine",
     page_icon="🧊",
 )

classes = ['person', 'bird', 'cat', 'cow', 'dog', 'horse', 'sheep', 'aeroplane', 'bicycle',
            'boat', 'bus', 'car', 'motorbike', 'train', 'bottle', 'chair', 'dining table',
            'potted plant', 'soda', 'tv/monitor']

st.title("Toy Image Search Engine")
st.write("Upload an image and you will get similar images in return.")
st.write("Similarity is determined by the distance of your image's intermediate layer activations of PyTorch's pretrained ResNet34 model to the PASCAL VOC 2007 dataset.")
class_expander = st.expander("The dataset is very small (~100 MB) due to heroku restrictions, so the engine will work best for any of these classes:")
class_expander.write(classes)

st.write("Thanks for checking it out!")
st.markdown("Made by Ishan Sharma  [LinkedIn](https://www.linkedin.com/in/ishanshar/) | [Github](https://github.com/ishandotsh)")

trfms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

model = resnet34(pretrained=True)
model.eval()

features = {}
def get_features(name):
    def hook(model, input, output):
        features[name] = output.detach()
    return hook
model.avgpool.register_forward_hook(get_features('data'))

data_dir = 'data'

@st.cache
def get_dataset():
    return datasets.ImageFolder(data_dir, None)

@st.cache
def load_ftrs():
    return np.load('stored_ftrs.npy', allow_pickle=True)


check_dataset = get_dataset()
stored_ftrs = load_ftrs()

def compare(compare_img_path):
    compare_ftrs = []
    with torch.no_grad():
        # img = Image.open(compare_img_path)
        img = compare_img_path
        img = trfms(img)
        img = img.unsqueeze(0)

        output = model(img)
        compare_ftrs.append(features['data'].cpu().numpy())
    return compare_ftrs

def get_distances(compare_img_tensor):
    distances = []
    for batch in stored_ftrs:
        for img in batch:
            dist = np.linalg.norm(compare_img_tensor - img)
            distances.append(dist)
    return distances

def get_best_indices(distances, n=10):
    dist_sorted = sorted(distances)
    image_idxs = []
    for d in dist_sorted[:n]:
        ind = distances.index(d)
        if ind not in image_idxs:
            image_idxs.append(ind)
    return image_idxs

def get_best_matches(image_idxs):
    results = {}
    for i in image_idxs:
        # print(check_dataset.samples[i])
        results[i] = check_dataset.samples[i][0]
    return results


def gen_random(n=5):
    rand_idxs = random.sample(range(len(check_dataset)), 5)
    rand_imgs_paths = []
    rand_imgs = []
    # print(f"Indexes: {rand_idxs}")
    for i in rand_idxs:
        rand_imgs_paths.append(check_dataset.samples[i][0])

    for img_path in rand_imgs_paths:
        img = Image.open(img_path)
        rand_imgs.append(img)

    return rand_imgs

def search(indx):
    image = rand_imgs[indx]
    ftrs = compare(image)
    ds = get_distances(ftrs[0])
    best_indxs = get_best_indices(ds, 11)
    best_indxs.pop(0)
    matches = get_best_matches(best_indxs)
    res_imgs = []
    for i in best_indxs:
        match_img = Image.open(matches[i])
        res_imgs.append(match_img)
    
    for i in range(2):
        cols = st.columns(4)
        cols[0].image(res_imgs[0 + i*4], use_column_width=True)
        cols[1].image(res_imgs[1 + i*4], use_column_width=True)
        cols[2].image(res_imgs[2 + i*4], use_column_width=True)
        cols[3].image(res_imgs[3 + i*4], use_column_width=True)

rand_imgs = gen_random()

st.header("Select from random selection:")

cols = st.columns([1, 6])
selected = cols[0].radio("Select your image.", range(1, 6))
cols[0].button("Generate New")
cols[1].image(rand_imgs[selected-1])
search(selected-1)

st.header("Or upload your own:")
file_up = st.file_uploader("Upload", type=["jpg", "png", "jpeg"])

if file_up is not None:

    file_details = {
        "FileName": file_up.name, 
        "FileType": file_up.type,
        "FileSize": file_up.size
    }

    image = Image.open(file_up)
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    ftrs = compare(image)
    ds = get_distances(ftrs[0])
    best_indxs = get_best_indices(ds)
    matches = get_best_matches(best_indxs)

    st.write("Similar images:")
    for i in best_indxs:
        match_img = Image.open(matches[i])
        st.image(match_img, use_column_width=True)