import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mlflow
import mlflow.azureml
import seaborn as sns
import argparse

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score, roc_curve

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

def prepareDataset(df):
    Y = df['ArrDelay15'].values
    synth_df = df.drop(columns=['ArrDelay15'])
    return synth_df, Y

def analyze_model(clf, X_test, Y_test, preds):
    with mlflow.start_run() as run:
        accuracy = accuracy_score(Y_test, preds)
        print(f'Accuracy', np.float(accuracy))
        mlflow.log_metric(f'Accuracy', np.float(accuracy))

        precision = precision_score(Y_test, preds, average="macro")
        print(f'Precision', np.float(precision))
        mlflow.log_metric(f'Precision', np.float(precision))
        
        recall = recall_score(Y_test, preds, average="macro")
        print(f'Recall', np.float(recall))
        mlflow.log_metric(f'Recall', np.float(recall))
        
        f1score = f1_score(Y_test, preds, average="macro")
        print(f'F1 Score', np.float(f1score))
        mlflow.log_metric(f'F1 Score', np.float(f1score))
        
        mlflow.sklearn.log_model(clf, artifact_path="outputs", registered_model_name="fd_model_mlflow_proj")
        
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
        fig.savefig("ConfusionMatrix.png")
        mlflow.log_artifact("ConfusionMatrix.png")
        plt.close()

        preds_proba = clf.predict_proba(X_test)[::,1]
        fpr, tpr, _ = roc_curve(Y_test, preds_proba, pos_label = clf.classes_[1])
        auc = roc_auc_score(Y_test, preds_proba)
        plt.plot(fpr, tpr, label="data 1, auc=" + str(auc))
        plt.legend(loc=4)
        plt.show()
        plt.close()

# az ml job create -f train.yml

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--data", type=str, help="input data path")

    args = parser.parse_args()
    print(args.data)

    data = pd.read_csv(args.data+'/flightdelayweather_ds_clean.csv')

    mlflow.sklearn.autolog()

    synth_df, Y = prepareDataset(data)

    #Split dataset
    X_train, X_test, Y_train, Y_test, A_train, A_test = split_dataset(synth_df, Y)

    # Setup scikit-learn pipeline
    numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])

    clf = Pipeline(steps=[('classifier', LogisticRegression(solver='liblinear', fit_intercept=True))])

    model = clf.fit(X_train, Y_train)
    preds = clf.predict(X_test)
    analyze_model(clf, X_test, Y_test, preds)
