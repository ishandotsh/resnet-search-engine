# Similar Image Search

Live: https://resnet-search-engine.herokuapp.com/

# Run on your machine

Run these commands to check it out

```shell

    git clone https://github.com/ishandotsh/resnet-search-engine.git
    pip3 install -r requirements.txt
    streamlit run app.py
```

You can add more images to the data folder. Create a new folder inside data, pytorch's dataset wrapper will handle the loading.
Be sure to run `python3 create_ftrs.py` if you do add more data.

# Description

Upload an image and you will get similar images in return.

Similarity is determined by the distance of your image's intermediate layer activations of PyTorch's pretrained ResNet34 model to the PASCAL VOC 2007 dataset.

The dataset is very small (~100 MB) due to heroku restrictions, so the engine will work best for any of these classes
['person', 'bird', 'cat', 'cow', 'dog', 'horse', 'sheep', 'aeroplane', 'bicycle', 'boat', 'bus', 'car', 'motorbike', 'train', 'bottle', 'chair', 'dining table','potted plant', 'soda', 'tv/monitor']

