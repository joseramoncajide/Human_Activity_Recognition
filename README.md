
Machine Learning and Human Activity Recognition: Building a Classifier for Wearable Accelerometers’ Data
===================
`#DataScience` `#RStats`  `#MachineLearning` `#HumanActivityRecognition`  `#Wearable` 


## Objetive ##

Implement a machine learning model to predict if body postures and movements when a unilateral dumbbell biceps curl repetitions are being realized.

Data was collected from accelerometers' used in devices such as Jawbone Up, Nike FuelBand, and Fitbit.

## About the methodology ##
The CRISP-DM (*cross-industry process for data mining*) methodology provides a structured approach to planning a data mining task and was applied to this project.

## Modelling

In order to classify if each of the repetitions were made according to the specification (Class A), throwing the elbows to the front (Class B), lifting the dumbbell only halfway (Class C), lowering the dumbbell only halfway (Class D) and throwing the hips to the front (Class E), **Random Forest** Machine Learning algorithm was applied to the 52 variables collected from the sensors.

The final model achieved **99.3% accuracy** on its classification task.

![Image from Weight Lifting Exercises Dataset](img/on-body-sensing-schema.png)

**Files**:

 1. [`har.md`](har.md): a markdown script with the project.
 2. [`har.Rmd`](har.Rmd): a R markdown script for reproducible research .
 3. [`solution.csv`](solution.csv): Labeled test data.


## About the authors

**Jose Ramón Cajide**
Digital Analyst at El Arte de Medir 

 - https://www.linkedin.com/in/jrcajide
 - [@jrcajide](https://twitter.com/jrcajide)


### More information
For more information about the Weight Lifting Exercises Dataset and the project visit: http://groupware.les.inf.puc-rio.br/har
