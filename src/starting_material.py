from sklearn.datasets import fetch_20newsgroups
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer


if __name__ == "__main__":
    # TODO: expose a nice CLI
    NEWSGROUP_DOWNLOAD_PATH = "./"
    binary_target = "talk.religion.misc", "alt.atheism", "soc.religion.christian"
    random_state = 123
    min_token_document_frequency = 10

    newsgroup_bunch_train = fetch_20newsgroups(
        data_home=NEWSGROUP_DOWNLOAD_PATH,
        random_state=random_state,
        subset="train",
        remove=("headers", "footers", "quotes"),
    )
    newsgroup_bunch_val = fetch_20newsgroups(
        data_home=NEWSGROUP_DOWNLOAD_PATH,
        random_state=random_state,
        subset="test",
        remove=("headers", "footers", "quotes"),
    )

    tfidf_transformer = TfidfVectorizer(
       min_df = min_token_document_frequency 
    )

    print("Fitting my tfidf transformer...")
    tfidf_transformer.fit(newsgroup_bunch_train.data)
    print("Fitting my tfidf transformer... Done")
    # TODO: add a scaler

    # TODO: we could use a sklearn.pipeline.Pipeline to chain feature
    # transform and classiication.
    print("Transforming train and test splits into tfidf values...")
    train_X = tfidf_transformer.transform(newsgroup_bunch_train.data)
    test_X = tfidf_transformer.transform(newsgroup_bunch_val.data)
    print("Transforming train and test splits into tfidf values... Done")

    # Prepare the target
    # TODO: factorize instead of copy/pasting for train and test
    target_train = 0
    target_test = 0
    for binary_target_ in binary_target:
        binary_target_idx = newsgroup_bunch_train.target_names.index(binary_target_)
        target_train |= newsgroup_bunch_train.target == binary_target_idx
        target_test |= newsgroup_bunch_val.target == binary_target_idx

    clf = DecisionTreeClassifier(random_state=random_state)
    print(f"Train {clf} classifier...")
    pred_train = clf.fit(train_X, target_train).predict(train_X)
    print(f"Train {clf} classifier... Done.")
    pred_test = clf.predict(test_X)
