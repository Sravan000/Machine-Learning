import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score 

def main():
    st.title("Binary Classification Web Application")
    st.sidebar.title("Binary classification Web Application")
    st.markdown("Are mushrooms edible or  poisionous?")
    st.sidebar.markdown("Are mushrooms edible or  poisionous?")

    @st.cache(persist = True)
    def load_data():
        data = pd.read_csv("/home/rhyme/Desktop/Project/mushrooms.csv")
        label = LabelEncoder()
        for col in data.columns:
            data[col] = label.fit_transform(data[col])

        return data

    df = load_data()

    @st.cache(persist = True)
    def split(df):
        y = df.type 
        x = df.drop(columns=['type'])
        x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=0)

        return x_train,x_test,y_train,y_test

    
    def plot_metrics(metrics_list):
        if "Confusion Matrix" in metrics_list:
            st.subheader("Confusion Matrix")
            plot_confusion_matrix(model,x_test,y_test,display_labels=class_names)
            st.pyplot()


        if "ROC Curve" in metrics_list:
            st.subheader("ROC Curve")
            plot_roc_curve(model,x_test,y_test)
            st.pyplot()

        if "Precision-Recal Curve" in metrics_list:
            st.subheader("Precisoin-Recall Curve")
            plot_precision_recall_curve(model,x_test,y_test)
            st.pyplot()


    df = load_data()

    x_train,x_test,y_train,y_test = split(df)
    class_names = ["edible","poisonous"]

    st.sidebar.subheader("Choose Classifier: ")
    classifier = st.sidebar.selectbox("classifier",("Support Vector Machine(SVM)","Logistic Regression","Random Forest"))

    if classifier == "Support Vector Machine(SVM)":
        st.sidebar.subheader("Model Hyperparameters")
        C = st.sidebar.number_input("C (Regularization Parameter)",0.01,10.0,step = 0.01,key='C')

        kernel = st.sidebar.radio("Kernel",("rbf","linear"),key='kernel')

        gamma = st.sidebar.radio("Gamma (Kernel Coeficient)",("scale","auto"),key = 'gamma')

        metrics =st.sidebar.multiselect("What Metrics to plot? ",("Confusion Matrix","ROC Curve","Precision-Recal Curve"))

        if st.sidebar.button("Classify",key='Classify'):
            st.subheader("Support Vector Machines(SVM)")
            model = SVC(C=C,kernel = kernel,gamma=gamma)
            model.fit(x_train,y_train)
            accuracy = model.score(x_test,y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy: ",accuracy.round(2))
            st.write("Precision: ",precision_score(y_test,y_pred,labels = class_names).round(2))
            st.write("Recall ",recall_score(y_test,y_pred,labels = class_names).round(2))

            plot_metrics(metrics)



    if st.sidebar.checkbox("Show raw data",False):
        st.subheader("Mushroom Data Set (Classification)","support Vector Machine(SVM)")
        st.write(df)



if __name__ == '__main__':
    main()


