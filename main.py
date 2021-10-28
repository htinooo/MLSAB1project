import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

st.set_option('deprecation.showPyplotGlobalUse', False)
plt.style.use("dark_background")


def test_split():
    X = data.iloc[:,:-1]
    y = data.iloc[:,-1]
    global X_train, X_test, y_train, y_test
    testsize = st.slider("Test Size(%)",5,40,20)
    randstate = st.radio("Random State", (0,None))
    from sklearn.model_selection import train_test_split
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=testsize/100,random_state=randstate)


def reg_algo():
    global model, y_pred, algo
    algo = st.selectbox("Select the regression method", ["LinearRegression", "DecisionTreeRegressor", "SupportVectorMachines"])    
    if algo == "DecisionTreeRegressor":
        from sklearn.tree import DecisionTreeRegressor
        model = DecisionTreeRegressor()
        model.fit(X_train,y_train)
        y_pred = model.predict(X_test)
    if algo == "LinearRegression":
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        model.fit(X_train,y_train)
        y_pred = model.predict(X_test)
    if algo == "SupportVectorMachines":
        from sklearn.svm import SVR
        model = SVR(kernel='linear')
        model.fit(X_train,y_train)
        y_pred = model.predict(X_test)


def class_algo():
    global model, y_pred
    algo = st.selectbox("Select the classification algorithm", ["DecisionTreeClassifier", "LogisticRegression", "SupportVectorMachines"])
    if algo == "DecisionTreeClassifier":
        from sklearn.tree import DecisionTreeClassifier
        model = DecisionTreeClassifier()
        model.fit(X_train,y_train)
        y_pred = model.predict(X_test)
    if algo == "LogisticRegression":
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression()
        model.fit(X_train,y_train)
        y_pred = model.predict(X_test)
    if algo == "SupportVectorMachines":
        from sklearn.svm import SVC
        model = SVC(kernel='linear')
        model.fit(X_train,y_train)
        y_pred = model.predict(X_test)


def reg_score():
    from sklearn.metrics import mean_absolute_error,mean_squared_error
    MAE = round(mean_absolute_error(y_test,y_pred),3)
    MSE = round(mean_squared_error(y_test,y_pred),3)
    RMSE = round(np.sqrt(MSE),3)
    st.info("Mean Absolute Error = " + str(MAE))
    st.info("Mean Squared Error = " + str(MSE))
    st.info("Root Mean Squared Error = " + str(RMSE))
    if algo == "LinearRegression":
        coef_df = pd.DataFrame(model.coef_,X_test.columns,columns=['Coefficient'])
        st.dataframe(coef_df)


def class_score():
    from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,classification_report
    accuracy = 'Accuracy:   '+ str(round(accuracy_score(y_test,y_pred)*100,2))+'%'
    st.info(str(accuracy))
    precision = 'Precision(macro avg):  '+ str(round(precision_score(y_test,y_pred,average='macro')*100,2))+'%'
    st.info(str(precision))
    recall = 'Recall(macro avg):    '+ str(round(recall_score(y_test,y_pred,average='macro')*100,2))+'%'
    st.info(str(recall))
    f1 = 'F1 Score(macro avg):  '+ str(round(f1_score(y_test,y_pred,average='macro')*100,2))+'%'
    st.info(str(f1))
    if st.checkbox("Show classification report"):
        report = pd.DataFrame(classification_report(y_test,y_pred,output_dict=True)).transpose()
        st.dataframe(report)


def conf_matrx():
    confusion_matrix = pd.crosstab(y_test,y_pred,rownames=['Actual'],colnames = ['Predicted'])
    sns.heatmap(confusion_matrix,annot=True,cmap="Reds")
    st.pyplot()


st.title("MLSAB1 Project")
st.header("Htin Oo (21htinoo99@gmail.com)")
dataset = st.selectbox("Choose a dataset", ["", "Advertising", "Health Insurance", "Fish", "Iris"])

if dataset == "Advertising":
    data = pd.read_csv("DATA/Advertising.csv")
    if st.checkbox("Show DataFrame"):
            st.dataframe(data)
    test_split()
    reg_algo()
    if st.checkbox("Show data visualization"):
        feat = st.selectbox("Select the feature", X_test.columns)
        fig,axes = plt.subplots(nrows=2,ncols=1,figsize=(7,12))
        sns.scatterplot(ax=axes[0],x=data[feat],y=data.iloc[:,-1])
        axes[0].set_title(f"Relationship between {data.columns[-1]} and {feat}")
        axes[1].plot(X_test[feat],y_test,'o', label=f"{data.columns[-1]}-test")
        axes[1].plot(X_test[feat],y_pred,'o',color='orange',label=f"{data.columns[-1]}-predict")
        axes[1].set_ylabel(data.columns[-1])
        axes[1].set_xlabel(feat)
        axes[1].legend()
        st.pyplot()
    if st.checkbox("Show model performance"):
        reg_score()
    if st.checkbox("Predict with customized data"):
        tv = st.slider("TV", data.TV.min(), data.TV.max(), data.TV.mean())
        radio = st.slider("Radio", data.radio.min(), data.radio.max(), data.radio.mean())
        news = st.slider("Newspaper", data.newspaper.min(), data.newspaper.max(), data.newspaper.mean())
        if st.button("Proceed"):
            pred = model.predict([[tv,radio,news]])
            st.success("Predicted sales = " + str(round((pred[0]),2)))


if dataset == "Health Insurance":
    df = pd.read_csv("DATA/insurance.csv")
    if st.checkbox("Show DataFrame"):
            st.dataframe(df)
    data = df.replace({"sex":{"female":0, "male":1}, "smoker":{"no":0, "yes":1}, 
            "region":{"southeast":0,"northeast":1,"southwest":2,"northwest":3}})
    test_split()
    reg_algo()
    if st.checkbox("Show data visualization"):
        feat = st.selectbox("Select the feature", X_test.columns)
        fig,axes = plt.subplots(nrows=2,ncols=1,figsize=(7,12))
        if feat == X_test.columns[0] or feat == X_test.columns[2]:
            sns.distplot(ax=axes[0],x=df[feat])
            sns.scatterplot(ax=axes[1],x=df[feat],y=data.iloc[:,-1])
            st.pyplot()
        else:
            sns.countplot(ax=axes[0],x=df[feat])
            sns.swarmplot(ax=axes[1],x=df[feat],y=data.iloc[:,-1])
            st.pyplot()
    if st.checkbox("Show model performance"):
        reg_score()
    if st.checkbox("Predict with customized data"):
        age = st.number_input("Age", data.age.min(), data.age.max(), step=1)
        sex0 = st.radio("Sex", ("Male","Female"))
        bmi = st.slider("BMI", data.bmi.min(), data.bmi.max(), data.bmi.mean())
        children = st.number_input("Childern", data.children.min(), data.children.max(), step=1)
        smoker0 = st.radio("Smoke", ("No","Yes"))
        region0 = st.radio("Region", ("Southeast","Northeast","Southwest","Northwest"))       
        sex = 1 if sex0 == "Male" else 0
        smoker = 0 if smoker0 == "No" else 1
        region = 0 if region0 == "Southeast" else 1 if region0 == "Northeast" else 2 if region0 == "Southwest" else 3
        if st.button("Proceed"):
            pred = model.predict([[age,sex,bmi,children,smoker,region]])
            st.success("Predicted charges = " + str(round((pred[0]),2)))


if dataset == "Iris":
    from sklearn import datasets
    iris = datasets.load_iris()
    data = pd.DataFrame(data=iris.data,columns= iris.feature_names)
    data['Species'] = iris.target
    if st.checkbox("Show DataFrame"):
        st.dataframe(data)
    test_split()
    class_algo()
    if st.checkbox("Show data visualization"):
        x_feat = st.selectbox("Feature on X", iris.feature_names)
        y_feat = st.selectbox("Feature on Y", iris.feature_names, 1)
        sns.scatterplot(data[x_feat], data[y_feat], hue=iris.target, palette="pastel")
        st.info("0 = Setosa, 1 = Versicolor, 2 = Virginica")
        st.pyplot()
    if st.checkbox("Show Confusion Matrix"):
        conf_matrx()
    if st.checkbox("Show Model Score"):
        class_score()
    if st.checkbox("Predict custom data"):
        sl = st.slider("Sepal Length (cm)",4.0,8.0,5.8)
        sw = st.slider("Sepal Width (cm)",2.0,4.5,3.0)
        pl =st.slider("Petal Length (cm)",1.0,7.0,4.4)
        pw = st.slider("Petal Width (cm)",0.1,2.5,1.0)
        if st.button("Proceed"):
            pred = model.predict(np.array([(sl,sw,pl,pw)]))
            name = iris.target_names[pred]
            st.success("Predicted iris = " + name[0])   


if dataset == "Fish":
    df = pd.read_csv("DATA/Fish.csv")
    if st.checkbox("Show DataFrame"):
        st.dataframe(df)
    data = df.replace({"Species":{"Bream":0,"Roach":1,"Whitefish":2,"Parkki":3,"Perch":4,"Pike":5,"Smelt":6}})
    test_split()
    class_algo()
    if st.checkbox("Show data visualization"):
        x_feat = st.selectbox("Feature on X", data.columns[0:-1])
        y_feat = st.selectbox("Feature on Y", data.columns[0:-1], 1)
        sns.scatterplot(df[x_feat], df[y_feat], hue=df.iloc[:,-1], palette="pastel")
        st.pyplot()
    if st.checkbox("Show Confusion Matrix"):
        conf_matrx()
    if st.checkbox("Show Model Score"):
        class_score()
    if st.checkbox("Predict custom data"):
        wt = st.slider("Weight", data.Weight.min(), data.Weight.max(), data.Weight.mean())
        l1 = st.slider("Length1", data.Length1.min(), data.Length1.max(), data.Length1.mean())
        l2 = st.slider("Length2", data.Length2.min(), data.Length2.max(), data.Length2.mean())
        l3 = st.slider("Length3", data.Length3.min(), data.Length3.max(), data.Length3.mean())
        h = st.slider("Height", data.Height.min(), data.Height.max(), data.Height.mean())
        wd = st.slider("Width", data.Width.min(), data.Width.max(), data.Width.mean())
        if st.button("Proceed"):
            pred = model.predict(np.array([(wt,l1,l2,l3,h,wd)]))
            if pred == 0:
                name = "Bream"
            if pred == 1:
                name = "Roach"
            if pred == 2:
                name = "Whitefish"
            if pred == 3:
                name = "Parkki"
            if pred == 4:
                name = "Perch"
            if pred == 5:
                name = "Pike"
            if pred == 6:
                name = "Smelt"
            st.success("Predicted iris = " + name)