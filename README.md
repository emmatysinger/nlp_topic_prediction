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
