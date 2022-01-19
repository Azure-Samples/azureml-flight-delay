from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import os
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score, roc_curve
from azureml.core.run import Run

def split_dataset(X_raw, Y):
    A = X_raw[['UniqueCarrier']]
    X = X_raw.drop(labels=['UniqueCarrier'],axis = 1)
    X = pd.get_dummies(X)


    le = LabelEncoder()
    Y = le.fit_transform(Y)

    X_train, X_test, Y_train, Y_test, A_train, A_test = train_test_split(X_raw, 
                                                        Y, 
                                                        A,
                                                        test_size = 0.2,
                                                        random_state=123,
                                                        stratify=Y)

    # Work around indexing bug
    X_train = X_train.reset_index(drop=True)
    A_train = A_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    A_test = A_test.reset_index(drop=True)

    return X_train, X_test, Y_train, Y_test, A_train, A_test 

def prepareDataset(X_raw):
    df = X_raw
    Y = df['ArrDelay15'].values
    synth_df = df.drop(columns=['ArrDelay15'])
    return synth_df, Y

def analyze_model(clf, X_test, Y_test, preds, run):
    accuracy = accuracy_score(Y_test, preds)
    print(f'Accuracy', np.float(accuracy))
    run.log('Accuracy', np.float(accuracy))

    precision = precision_score(Y_test, preds, average="macro")
    print(f'Precision', np.float(precision))
    run.log('Precision', np.float(precision))

    recall = recall_score(Y_test, preds, average="macro")
    print(f'Recall', np.float(recall))
    run.log('Recall', np.float(recall))

    f1score = f1_score(Y_test, preds, average="macro")
    print(f'F1 Score', np.float(f1score))
    run.log('F1 Score', np.float(f1score))

    class_names = clf.classes_
    fig, ax = plt.subplots()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)
    sns.heatmap(pd.DataFrame(confusion_matrix(Y_test, preds)), annot=True, cmap='YlGnBu', fmt='g')
    ax.xaxis.set_label_position('top')
    plt.tight_layout()
    plt.title('Confusion Matrix', y=1.1)
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.show()
    run.log_image('Confusion Matrix', plot=plt)
    plt.close()

    preds_proba = clf.predict_proba(X_test)[::,1]
    fpr, tpr, _ = roc_curve(Y_test, preds_proba, pos_label = clf.classes_[1])
    auc = roc_auc_score(Y_test, preds_proba)
    plt.plot(fpr, tpr, label="data 1, auc=" + str(auc))
    plt.legend(loc=4)
    plt.show()
    run.log_image('ROC Curve', plot=plt)
    plt.close()


def main():
    run = Run.get_context()
    # Fetch dataset from the run by name
    dataset = pd.read_csv('./flightdelayweather_ds_clean.csv')
    synth_df, Y = prepareDataset(dataset)

    #Split dataset
    X_train, X_test, Y_train, Y_test, A_train, A_test = split_dataset(synth_df, Y)

    # Setup scikit-learn pipeline
    numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])

    clf = Pipeline(steps=[('classifier', LogisticRegression(solver='liblinear', fit_intercept=True))])

    model = clf.fit(X_train, Y_train)
    preds = clf.predict(X_test)
    analyze_model(clf, X_test, Y_test, preds, run)

    os.makedirs('outputs', exist_ok=True)
    # files saved in the "outputs" folder are automatically uploaded into run history
    joblib.dump(model, 'outputs/model.joblib')
    

if __name__ == '__main__':
    main()
