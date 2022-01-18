# Applied Deep Learning
## Assignment 3
### Lukas Prenner
------------------------------------

# Description of Project
The aim of the project is to implement an existing deep learning model which should be able to successfully classify the faces of Sebastian Kurz and Alexander Schallenberg. The application to use the model will be a (local) web app with a simple interface where users will be able to upload an image of either one of the two individuals and get an answer of which of the two it actually is - including the probability of the prediction.

## Target Metric: Accuracy
Accuracy has been defined as the target metric to optimize for from the beginning. After several hours of setting up and debugging the training pipeline, however, the results from all internal tests have been practically 100% accurate. Moreover, the probability of each prediction is also mostly around 95% which shows how solid the FaceNet model works and how little room for optimization is left. (Note: the prediction probabilities and overall results might decrease with increasing number of classes.) Therefore, not a lot of optimization potential can be seen regarding the performance of the model using this data.

## Work Breakdown Structure
- Data Collection --> 4.5 hours (including manually resizing and filtering)
- First Implementation --> 18-20 hours
- Training and Tuning --> 3-4 hours
- Online Application --> 5
- Final Report --> 3.5 hours
- Preparing Presentation --> 4 hours
- Presentation --> 0.15 hours

## How to run the code
In order to run the current state of the code, check the requirements.txt and run:
```
pip install -r requirements.txt
```

To run the training-pipeline, run:
```
python3 Training-Pipeline.py
```

To run the actual Flask application, run:
```
python3 Kurz-or-Schallenberg.py
```

If any problems arise which are not covered by the instructions above, I am of course happy to support whenever I can.
Please just contact me via: lukas.prenner@student.tuwien.ac.at
