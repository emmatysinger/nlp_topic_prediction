# NLP and ML exercise: topic prediction.

This exercise is a case study on topic prediction using NLP. It contains
several sections that browse over several different skillsets. Feel free to
prioritize steps on which you are more at ease with and drop those that are
less inspiring.

The goal of this exercise is to build and/or analyze the performance of a
classifier for topic detection in the newsgroups 20 dataset.

We focus on a binary classification task, where the positive class matches
mails that discuss religion.

The classifier must be a single decision tree (using scikit-learn
implementation), trained on tf-idf features.

Such a pipeline is given as a starting point in the script
`starting_material.py`.

## Performance analysis

_What metrics could be relevant to assess the performance of our classifier ?_

### Relevant Metrics to Assess Classifier Performance

1. **Accuracy**: The proportion of correctly classified instances out of the total instances

2. **Precision**: Shows the classifier’s ability to avoid false positives.

3. **Recall**: Shows the classifier’s ability to capture all true positives.

4. **F1 Score**: Balances both precision and recall and is particularly useful since the dataset is imbalanced.

5. **Confusion Matrix**: Visual evaluation of the classifier.

_What can we say about the performance of the classifier we just trained for
this task ?_

### Analysis of the Classifier's Performance

Here are the results:

- **Accuracy**: 0.9082580987785449
- **Precision**: 0.638085742771685
- **Recall**: 0.6611570247933884
- **F1 Score**: 0.6494165398274987

1. An accuracy of ~91% shows that the classifier is generally performing well in terms of overall correct classifications, but because the dataset is imbalanced this is misleading. 

2. A precision of ~64% means that when the classifier predicts a positive label, it is correct 64% of the time.

3. A recall of ~66% means that the classifier is able to identify 66% of the actual positive instances.

4. The F1 score of ~65% provides a balance between precision and recall.

**Final Insight** – The classifier's high accuracy suggests it performs well overall, but the lower precision and recall indicate room for improvement in identifying positive cases.

Add all the code that you wish, that can support this analysis.

## Explainability

For a given sample for which our pipeline predicts a positive label, we would
like to answer the following question: what are the features that were the most
impactful in the prediction outcome ? Put differently, we would like to be able
to explain the predictions of our pipeline at the sample level, as a distribution
of importance over all the input features for a given sample.

For this part, you can take inspiration, or even just implement, the approach
suggested at http://blog.datadive.net/interpreting-random-forests .

Here is a suggested roadmap:

Using [relevant scikit-learn
documentation](https://scikit-learn.org/stable/auto_examples/tree/plot_unveil_tree_structure.html):

- Write a function that retrieves, for a single given sample that has been
  predicted to be in the positive class by the classifier, the list of features
  that are used under the hood by the decision tree to make this prediction.
  Using metadata from the TFIDF pipeline, the features must be mapped to the
  string of the token that they encode.

- Using the metadata available for all nodes of the decision tree, propose and
  implement a function that returns, as a float, an evaluation of how much a
  given feature impacted the prediction in the positive class. (tip: for
  instance, substract the normalized distribution of samples in the training
  set, before and after the corresponding nodes for this feature)

- Implement a function that use this value to filter most important feature for
  a given sample. Print a few examples.

### Explainability function
To calculate the explainability, I followed the suggested roadmap. First, the function retrieves the path of features traversed by the decision tree for a given sample. Then for each feature in this path, the importance score was calaulated. This was the difference between the impurity score for the current feature and the previous feature, which is a way of calculating how a feature helps in seperating the positive and negative samples. Finally, the features are ranked by their respective importance scores and the top 10 are returned. 

**Example 1**
Important features for sample: It is important if Christianity is being damaged by it. If
people who "speak in tongues" make claims that they are
miraculously speaking a foreign language through the power of
the Holy Spirit, when it can easily be shown that they are simply
making noises, it damages all Christians, since many who are
not Christians do not distinguish between the various sects.

The more modest claim for "tongues" that it is simply
uncontrolled praise in which "words fail you" is surely the one
that should be used by those who make use of this practice.

I agree with the point that "Charismatic" practices like this
can lead to forms of worship which are more about the
worshipper showing off than genuine praise for God; one of the
things Jesus warned us about.

Feature: god, Importance: 0.0541
Feature: win, Importance: 0.0168
Feature: government, Importance: 0.0161
Feature: gun, Importance: 0.0145
Feature: look, Importance: 0.0118
Feature: sci, Importance: 0.0100
Feature: crap, Importance: 0.0098
Feature: memory, Importance: 0.0096
Feature: 50, Importance: 0.0095
Feature: services, Importance: 0.0092

**Example 2**
Important features for sample: Subject: RE: Europe vs. Muslim Bosnians
From: f54oguocha
Date: 13 MAY 93 02:28:53 GMT
Serbs.
1929).
Islam 
not
who
Croats? 
has 
childhood was 

You've asked a crucial question that underlies much of the genocide. 
Bosnian Muslims are slavic in ethnicity. They speak Serbo-Croatian. But
there is a Christo-Slavic ideology whereby all true slavs are Christian
and anyone who converted to Islam thereby must have changed ethnicity by
changing religion.  See the poems of Ngegos or the novels of Ivo Andric
who brilliantly displays these attitudes on the part of what he calls
"the people" (i.e. Christian slavs).  For this reason, the war-criminals
call all the Bosnian Muslims "Turks" even though they are not ethnically
Turk and do not speak Turkish as their first language.  For this reason,
what is actually a genocide labeled against those who are ethnically
identical but religiously "other" is called, paradoxically, "ethnic
cleansing" rather than "religious cleansing."

Thus, while a war rages between Serbs and Croats as a continuation of
WWII, and older agenda, the annihilation of Islam and Muslims from
Bosnian, is being carried out under the cover of the Serbo-Croat war.

Regards,

Mike.
Feature: church, Importance: 0.0731
Feature: freedom religion, Importance: 0.0646
Feature: inside, Importance: 0.0627
Feature: right, Importance: 0.0572
Feature: god, Importance: 0.0541
Feature: interesting, Importance: 0.0353
Feature: 1920, Importance: 0.0298
Feature: bible, Importance: 0.0141
Feature: jesus, Importance: 0.0106
Feature: religion, Importance: 0.0084

**Example 3**
Important features for sample: 
The biblical arguments against homosexuality are weak at best, yet
Christ is quite clear about our obligations to the poor. How as 
Christians can we demand celibacy from homosexuals when we walk
by homeless people and ignore the pleas for help? 
Christ is quite clear on our obligations to the poor.

Thought for the day:

MAT 7:3 And why beholdest thou the mote that is in thy brother's eye, but
considerest not the beam that is in thine own eye? Or how wilt thou say to
thy brother, Let me pull out the mote out of thine eye; and, behold, a beam
is in thine own eye?  
Feature: adam, Importance: 0.0689
Feature: flaming, Importance: 0.0619
Feature: god, Importance: 0.0541
Feature: bible, Importance: 0.0141
Feature: jesus, Importance: 0.0106
Feature: religion, Importance: 0.0084
Feature: church, Importance: 0.0054
Feature: morality, Importance: 0.0053
Feature: christians, Importance: 0.0042


## Improving our pipeline

What could we improve accross the board to make a better, more useful pipeline
?  Give a few example, and if you wish, implement some alternative or
complementary strategies and compare them with the current baseline.

### Model Improvements: 
1. Remove stop words and use n-grams to add more context to the features.
   - This improved the F1-score to 0.669 (from 0.649)
2. Perform hyperparameter to optimize the model parameters.
   - This improved the F1-score to 0.686
   - Optimal parameters: max_depth = 20 and min_split = 5
3. Use ensemble methods such as random forest which includes multiple decision trees
   - This improved the F1-score to 0.739
   - Optimal parameters: max_depth = None, min_split = 5, num_estimators = 100
  
**Final insight** – These 3 improvements focus on the data preprocessing and optimizing the classifier model we use. In total the F1 score increased by 0.09 percentage points.

## Setting up a web service

Using the libraries of your choice, implement a web applications exposing
a route for submitting a sample, and another route to get the content, the
prediction, and the explanations, for of the latest submitted sample.

The web service uses Flask for the backend and a basic HTML landing page. To use the web service run and open in the url in a web browser:
```
python app.py
```
