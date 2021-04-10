# Song-Recommendation-using-Facial-Emotion-Recognition
The human face is an important part of an individual’s body and it particularly plays a significant role in knowing a person’s state of mind. This eliminates the dreary and tedious task of manually isolating or grouping songs into various records and helps in generating an appropriate playlist based on an individual’s emotional features.

People tend to listen to music based on their mood and interests. One can create an application to suggest songs for users based on their mood by capturing facial expressions.

Computer vision is an interdisciplinary field that helps convey a high-level understanding of digital images or videos to computers. Computer vision components can be used to determine the user’s emotion through facial expressions.

## Requirements
The project uses python 3.8, tensorflow, keras, open-cv, pandas, numpy, youtube-dl.

## Facial Emotion Recognition (FER) using Convolution Neural Network (CNN)
Generally, there are seven emotions on human face - Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise. 
In this Machine Learning model, FER 2013 dataset was used. The dataset was divided into 2 parts
1. Training Set - Containing 80% of the dataset.
2. Test Set - Containing 20% of the dataset.

The model was trained using convolution neural networks with activation function as relu. The performance was judged using 50 epochs and batch size 64, however the validation accuracy was saturated after 25 epochs.

The accuracy achieved on the training set was 95.12% and on the test set was 64.81%. The model was saved as "model.h5".

## Integration of the model with Captured Face Image using Haarcascades
The Face was detected in the image using Haarcascades. The image was cropped in the facial region. The cropped image was passed to the trained model and corresponding emotion was predicted by the model.

## Recommendation and Download of Song
Using the predicted emotion, the program recommends ten songs from the database. The songs are then downloaded in the "Songs" directory. Youtube-dl was used for downloading songs.

To run the program, execute the following command,
```
python main.py
```

## Contributors
1. Ritik Gupta
2. Vishu Garg

