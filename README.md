# dsan6600-final-project
Predicting traffic flow to identify the root causes of traffic in a Barbados roundabout

![Roundabout schematic](Roundabout.png)

## Helping Barbados

![Barbados Flag](BarbadosFlag.svg)

On the small island state of Barbados, cars, buses and taxis are the primary mode of transport, and traffic is a well-known problem that affects every Barbadian citizen. The Ministry of Transport and Works is trying to solve the problem with machine learning through an open competition.

Our task is to predict traffic congestion using machine learning, with the aim of recognising the root causes of traffic in a specific roundabout in Barbados. We have been provided with four streams of video data, labelled with the congestion rating for the entrance and exit timestamps, and our model predicts traffic congestion five minutes into the future. Ultimately, we are interested in identifying the root causes of increased time spent in the roundabout by developing features from unstructured video data. 

The model will potentially help the Ministry of Transport and Works predict traffic flow and identify and address root causes of traffic in roundabouts. They will use this information to design interventions to reduce traffic across the island, improving the lives of every citizen.

## Evaluation

This challenge uses multi-metric evaluation. There are two error metrics: F1 and Accuracy.

Your score on the leaderboard is the weighted mean of two metrics:

- Macro-F1 (70%): measures how well your model performs across all four congestion classes, treating each class equally. This is important because some classes may appear less often in the data.
- Accuracy (30%) - measures the overall percentage of correct predictions across all samples.

Our model aims for high accuracy and balanced performance across all classes.

For every row in the dataset, the submission contain 3 columns: id, Target and Target_Accuracy.

F1 is calculated from the column Target.

Accuracy is calculated from the column Target_Accuracy.
