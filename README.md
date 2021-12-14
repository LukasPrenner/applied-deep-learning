# Applied Deep Learning
## Assignment 2: Hacking
### Lukas Prenner
------------------------------------

# Description of Project
The aim of the project is to implement an existing deep learning model which should be able to successfully classify the faces of Sebastian Kurz and Alexander Schallenberg. The application to use the model will be a (local) web app with a simple interface where users will be able to upload an image of either one of the two individuals and get an answer of which of the two it actually is - including the probability of the prediction.

# Progress so far
As of now, the training pipeline as well as the prediction pipeline have been successfully implemented using python 3.7. The training pipeline uses the MTCNN package's detector to isolate faces from the images and then uses the FaceNet model to retrieve the relevant embeddings from the faces which are subsequently used to train a support vector classifier (SVC) which ultimately predicts the classes. The prediction pipeline takes the one single image placed in the folder '04_Prediction-Data', loads all the models and encoders created in the training pipeline and outputs the final class prediction of the model.

## Target Metric: Accuracy
Accuracy has been defined as the target metric to optimize for from the beginning. After several hours of setting up and debugging the training pipeline, however, the results from all internal tests have been practically 100% accurate. Moreover, the probability of each prediction is also mostly around 95% which shows how solid the FaceNet model works and how little room for optimization is left. (Note: the prediction probabilities and overall results might decrease with increasing number of classes.) Therefore, not a lot of optimization potential can be seen regarding the performance of the model using this data.

## Work Breakdown Structure
- Data Collection --> 4.5 hours (including manually resizing and filtering)
- First Implementation --> 15-20 hours
- Training and Tuning --> 3-4 hours
- Online Application --> 3 hours so far
- Final Report --> tbd.
- Preparing Presentation --> tbd.
- Presentation --> tbd.

## How to run the code
In order to run the current state of the code, several requirements have to be met:
- Python Version 3.7 must be used
- All packages from the 'Prerequisites.sh' file need to be installed (note: correct version!)

For running the prediction pipeline:
- One single image must be placed into the folder '04_Prediction-Data'

If any problems arise which are not covered by the instructions above, I am of course happy to support whenever I can.
Please just contact me via: lukas.prenner@student.tuwien.ac.at

# What's next
If no major bugs or feedback about potential errors arise, the focus of the next few weeks will be on implementing a simple (local) web app for uploading an image to be classified and visualizing the results to the user. After that, the final report as well as the presentation will be created.
