import dash as dash
from dash import dcc,dash_table
from dash import html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
# from plotly.subplots import make_subplots
# from datetime import date
import random
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.preprocessing import LabelEncoder
#from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.svm import SVC, SVR
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, confusion_matrix, mean_absolute_error, classification_report, r2_score
from scipy.stats import uniform, randint, uniform, loguniform
style={'textAlign':'center'}
steps=0.1
random_state= 42
marks= lambda min,max:{i:f"{i}" for i in range(min,max)}
models= ['LinearRegression','LogisticRegression','RandomForestRegressor','RandomForestClassifier',
         'GradientBoostingRegressor','GradientBoostingClassifier','SVM Regression',
         'SVM Classification', 'Naive Bayes Classifier','KNeighborsClassifier','KNeighborsRegressor',
         'Ridge Regression','Lasso Regression','ElasticNet Regression']
#===========================================
# Data cleaner functions
def remove_outliers(df,column):
    if df[column].dtype == 'int' or df[column].dtype == 'float':
        q1= df[column].quantile(0.25)
        q3= df[column].quantile(0.75)
        iqr= q3-q1
        lb= q1-1.5*iqr
        ub= q3-1.5*iqr
        df_new= df[(df[column]>= lb) & (df[column]<=ub)]
    return df_new

def id_checker(df, dtype='float'):
    """
    The identifier checker

    Parameters
    ----------
    df : dataframe
    dtype : the data type identifiers cannot have, 'float' by default
            i.e., if a feature has this data type, it cannot be an identifier

    Returns
    ----------
    The dataframe of identifiers
    """

    # Get the dataframe of identifiers
    df_id = df[[var for var in df.columns
                # If the data type is not dtype
                if (df[var].dtype != dtype
                    # If the value is unique for each sample
                    and df[var].nunique(dropna=True) == df[var].notnull().sum())]]
    # df_train.drop(columns=np.intersect1d(df_id.columns, df_train.columns), inplace=True)

    return df_id


def cleaner(df):
    df= df.drop_duplicates()
    df.columns= (df.columns.str.strip().str.lower().str.replace(" ","_").str.replace(
        "[()$@!#%^&*-=+~`/\|]","", regex=True
    ))

    df= df.dropna(axis='columns', how='all')
    df= df.dropna(how='all') #by rows on default

    for col in df.columns:
        if 'date' in col or 'time' in col:
            try:
                df[col]= pd.to_datetime(df[col])
            except:
                pass

        if df[col].isna().sum().all():
            if df[col].dtype == 'object':
                df[col].fillna(df[col].mode().max(), inplace=True)
            elif df[col].dtype == 'int':
                df[col].fillna(round(df[col].mean()), inplace=True)
            elif df[col].dtype == 'float':
                df[col].fillna(np.round(df[col].mean(),2), inplace=True)
            else:
                pass
        else:
            pass

    return df

def preprocess(df, target, features):
    if target in features:
        df= df[features]
    else:
        features += [target]
        df = df[features]

    df = cleaner(df)
    identifiers = id_checker(df).columns
    df = df.drop(columns=identifiers)
    tag_type= df[target].dtype
    if df[target].dtype == 'object':
        tsf= LabelEncoder().fit(df[target])
        target_column= tsf.transform(df[target])
    else:
        target_column= df[target]
        tsf= None

    df= df.drop(columns=target)

    numerical_columns= df.select_dtypes(include=['float64','int64']).columns
    categorical_columns= df.select_dtypes(include=['object']).columns

    for i in categorical_columns:
        df[i]= LabelEncoder().fit_transform(df[i])

    model_cols= numerical_columns.append(categorical_columns)
    X = df[model_cols]
    Y = target_column
    if tag_type == 'object':
        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.15,stratify=Y, random_state=random_state)
    else:
        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.15, random_state=random_state)
    sc = StandardScaler().fit(x_train)
    x_train = sc.transform(x_train)
    x_test = sc.transform(x_test)

    return tsf, sc, x_train, x_test, y_train, y_test

#===========================================

my_app= dash.Dash(__name__,external_stylesheets=[dbc.themes.MORPH]) #dbc.themes.MORPH | dbc.themes.SOLAR
server = my_app.server
my_app.layout= html.Div([
    html.P(""),
    html.H1("Data Analyzer"),
    html.P("by: Osemekhian Solomon Ehilen"),html.Br(),
    dcc.Markdown("""This Web App Collects Your ***Data(link)*** 
                    then Generates Your Analysis & Model"""),
    dbc.Input(id="in1",placeholder="Paste Link to Your Data...",type='text', size="lg", className="mb-3"),
    html.Pre('For example: https://raw.githubusercontent.com/datasciencedojo/datasets/refs/heads/master/titanic.csv'),
    dcc.RadioItems(options=['Raw Data', 'Cleaned Data'], value='Raw Data', id='radio'),
    html.Button("Download Cleaned Data",id='btn_csv'),html.Br(),
    dcc.Download(id='download-data-cleaned'), html.Br(),
    html.P('Remove outliers using one numerical column:'),
    dcc.Dropdown(id='drp', multi=False),
    dcc.Checklist(options=['Remove Outliers'], id='outlier'),
    html.Div(id="out1"),
    html.Br(),html.Div(id='outy'),
    html.Div(id='out2'),
    dcc.Store(id='store1'),html.Br(),

    html.H4("Univariate Charts"), html.Br(),
    html.P("Select Column:"),
    dcc.Dropdown(id='unidrp',multi=False),
    html.Div(id='uni'), html.Br(),

    html.H4("Bivariate Charts"), html.Br(),
    html.P('NB:Tweak the column selections to get some charts as desired'),
    html.P("Select Columns (2 max):"),
    dcc.Dropdown(id='bidrp',multi=True),
    html.Div(id='Bi'),

    html.Div(id='out3'), html.Br(),


    html.H4("Modeling"),
    html.P("Please select feature columns excluding target which is selected above:"),
    dcc.Dropdown(id='mod2drp',multi=True,placeholder="select features only..."),
    html.P("Please select target variable/column:"),
    dcc.Dropdown(id='moddrp',multi=False, placeholder="select target only..."),
    html.P("Please select model:"),
    dcc.Dropdown(id='modeldrp',options= [{'label':i,'value':i} for i in models]),
    html.Button("Analyze",id='btn_analyze'), html.Br(),html.Br(),

    html.Div(id='modelout'),
    html.Br(),html.Br(),
    dcc.Markdown("Bravelion | 2025 | Contact for Analysis: [email](mailto:oseme781227@gmail.com) ")
], style=style)

@my_app.callback([Output('out1','children'),
                        Output('outy','children'),
                        Output('out2','children'),
                        Output('store1', 'data'),
                        Output('unidrp','options'),
                        Output('bidrp','options'),
                      Output('drp','options'),
                      Output('moddrp','options'),
                      Output('mod2drp','options')],
                    [Input('in1','value'),
                     Input('radio','value'),
                     Input('outlier','value'),
                     Input('drp','value')]
                )
def link(text, rad, button, drpval):
    if text==None:
        return ""
    try:
        if button:
            data= pd.read_csv(text,low_memory=False)
            data.columns = (data.columns.str.strip().str.lower().str.replace(" ", "_").str.replace(
                "[()$@!#%^&*-=+~`/\|]", "", regex=True))

            if rad== 'Cleaned Data':
                data= cleaner(data)
                if data[drpval].dtype == 'object':
                    pass
                else:
                    data= remove_outliers(data, drpval)
            else:
                data = remove_outliers(data, drpval)
        else:
            data = pd.read_csv(text, low_memory=False)
            data.columns = (data.columns.str.strip().str.lower().str.replace(" ", "_").str.replace(
                "[()$@!#%^&*-=+~`/\|]", "", regex=True))

        if rad == 'Raw Data':
            df= data
            options = [{'label': col, 'value': col} for col in df.columns]
            desc = df.describe()
            desc.insert(0, 'stat', df.describe().index)
            return (html.Div(dash_table.DataTable(df.to_dict('records'),
                                                  [{"name": i, "id": i} for i in df.columns],
                                                  style_cell={'textAlign': 'left', 'padding': '5px'},
                                                  style_header={'backgroundColor': 'rgb(220, 220, 220)',
                                                                'fontWeight': 'bold'},
                                                  page_size=10,
                                                  style_data={"overflow": "hidden", "textOverflow": "ellipsis",
                                                              "maxWidth": 0})), html.H3("Data Description"),
                    html.Div(dash_table.DataTable(desc.to_dict('records'),
                                                  [{"name": i, "id": i} for i in desc.columns],
                                                  style_cell={'textAlign': 'left', 'padding': '5px'},
                                                  style_header={'backgroundColor': 'rgb(220, 220, 220)',
                                                                'fontWeight': 'bold'},
                                                  page_size=20,
                                                  style_data={"overflow": "hidden", "textOverflow": "ellipsis",
                                                              "maxWidth": 0})),
                    df.to_json(orient='table'), options, options, options,options,options)
        else:
            df= cleaner(data)
            options = [{'label': col, 'value': col} for col in df.columns]
            desc = df.describe()
            desc.insert(0, 'stat', df.describe().index)
            return (html.Div(dash_table.DataTable(df.to_dict('records'),
                                          [{"name": i, "id": i} for i in df.columns],
                                          style_cell={'textAlign': 'left','padding':'5px'},
                                          style_header={'backgroundColor':  'rgb(220, 220, 220)','fontWeight': 'bold'},
                                                 page_size=10, style_data={"overflow":"hidden","textOverflow":"ellipsis",
                                                                           "maxWidth":0})),html.H3("Data Description"),html.Div(dash_table.DataTable(desc.to_dict('records'),
                                          [{"name": i, "id": i} for i in desc.columns],
                                          style_cell={'textAlign': 'left','padding':'5px'},
                                          style_header={'backgroundColor':  'rgb(220, 220, 220)','fontWeight': 'bold'},
                                                 page_size=20, style_data={"overflow":"hidden","textOverflow":"ellipsis",
                                                                           "maxWidth":0})),
                    df.to_json(orient='table'), options,options, options, options,options)
    except:
        return ""

@my_app.callback(Output('uni','children'),
                 [Input('store1','data'),
                  Input('unidrp','value'),
                  Input('radio','value'),
                  Input('outlier','value')])
def chart(data, column, rad, button):
    try:
        if button:
            pass
        if rad:
            df = pd.read_json(data, orient='table')
            fig = px.histogram(df, x=column, title=f"Histogram Plot for {column}")
            fig2 = px.box(df, y=column, title=f" Box Plot for {column}")
            fig3 = px.line(df, y=column, title=f" Line Plot for {column}")

            return html.Div([dcc.Graph(figure=fig),dcc.Graph(figure=fig2),dcc.Graph(figure=fig3)])
    except:
        return ""

@my_app.callback(Output('Bi','children'),
                 [Input('store1','data'),
                  Input('bidrp','value'),
                  Input('radio', 'value'),
                  Input('outlier', 'value')])
def charts(data, columns, rad, button):
    try:
        if button:
            pass
        if rad:
            columns= columns[:2]
            df = pd.read_json(data, orient='table')
            cor= df.corr()
            fig2 = px.box(df, x=columns[0], y=columns[-1], title=f"Box Plot for {columns[0]} & {columns[-1]}") #points='all',
            fig3 = px.scatter(df, x=columns[0], y=columns[-1], title=f"Scatter Plot for {columns[0]} & {columns[-1]}")
            fig4= px.imshow(cor, text_auto=True, title=f"Heatmap Correlation Plot",aspect="auto")
            fig5 = px.pie(df, values=columns[0], names=columns[-1], hole=0.1, title=f"Pie Plot for {columns[0]} & {columns[-1]}")
            fig = px.bar(df, x=columns[0], y=columns[-1], title=f"Bar Plot for {columns[0]} & {columns[-1]}")

            return html.Div([dcc.Graph(figure=fig2),dcc.Graph(figure=fig3),dcc.Graph(figure=fig4),dcc.Graph(figure=fig),dcc.Graph(figure=fig5)])
    except:
        return ""
#======================================================================================
@my_app.callback(Output('modelout','children'),
                 [Input('store1','data'),
                  Input('modeldrp','value'),
                  Input('moddrp', 'value'),
                  Input('mod2drp','value'),
                  Input('btn_analyze', 'n_clicks')])
def out1( data, model_type, target, features, btn_analyze):
    try:
        if btn_analyze:
            df= pd.read_json(data, orient='table')
            label_transform, standard_scaler, x_train, x_test, y_train, y_test = preprocess(df,target,features)

            if df[target].dtype == 'object':
                if model_type == 'LogisticRegression':
                    model= LogisticRegression(random_state=random_state)
                    param_dist = {
                        'C': uniform(0.001, 1000),  # Uniform distribution for C
                        'penalty': ['l1', 'l2'],  # Regularization type
                        'solver': ['liblinear', 'saga'],  # Solvers that support L1 and L2
                        'max_iter': randint(50, 500),  # Random integer between 50 and 500
                        'class_weight': [None, 'balanced'],  # Handle class imbalance
                        'fit_intercept': [True, False],  # Whether to fit intercept
                        'tol': uniform(1e-4, 1e-2),  # Tolerance for stopping criteria
                    }
                    # Initialize RandomizedSearchCV
                    random_search = RandomizedSearchCV(
                        estimator=model,
                        param_distributions=param_dist,
                        n_iter=30,  # Number of parameter settings to sample
                        scoring='accuracy',  # Metric to evaluate
                        cv=5,  # Number of cross-validation folds
                        random_state=42,
                        n_jobs=-1,  # Use all available CPU cores
                    )
                elif model_type =='RandomForestClassifier':
                    model= RandomForestClassifier(random_state=random_state)
                    param_dist = {
                        'n_estimators': randint(50, 500),  # Number of trees
                        'max_depth': [None] + list(randint(10, 50).rvs(5)),  # Maximum depth
                        'min_samples_split': randint(2, 20),  # Minimum samples to split
                        'min_samples_leaf': randint(1, 10),  # Minimum samples at leaf
                        'max_features': ['auto', 'sqrt', 'log2', 0.5],  # Features to consider
                        'bootstrap': [True, False],  # Bootstrap sampling
                        'criterion': ['gini', 'entropy'],  # Split criterion
                        'max_leaf_nodes': [None] + list(randint(10, 50).rvs(5)),  # Maximum leaf nodes
                        'min_impurity_decrease': uniform(0.0, 0.2),  # Impurity decrease
                        'class_weight': [None, 'balanced'],  # Class weights
                    }
                    # Initialize RandomizedSearchCV
                    random_search = RandomizedSearchCV(
                        estimator=model,
                        param_distributions=param_dist,
                        n_iter=30,  # Number of parameter settings to sample
                        scoring='accuracy',  # Metric to evaluate
                        cv=5,  # Number of cross-validation folds
                        random_state=42,
                        n_jobs=-1,  # Use all available CPU cores
                    )
                elif model_type == 'GradientBoostingClassifier':
                    model= GradientBoostingClassifier(random_state=random_state)
                    param_dist = {
                        'n_estimators': randint(50, 500),  # Number of boosting stages
                        'learning_rate': uniform(0.001, 0.2),  # Learning rate
                        'max_depth': randint(3, 15),  # Maximum depth of trees
                        'min_samples_split': randint(2, 20),  # Minimum samples to split
                        'min_samples_leaf': randint(1, 10),  # Minimum samples at leaf
                        'max_features': ['auto', 'sqrt', 'log2', 0.5],  # Features to consider
                        'subsample': uniform(0.5, 0.5),  # Fraction of samples for fitting
                        'loss': ['deviance', 'exponential'],  # Loss function
                        'criterion': ['friedman_mse', 'mse'],  # Split criterion
                        'min_impurity_decrease': uniform(0.0, 0.2),  # Impurity decrease
                        'warm_start': [True, False],  # Warm start
                    }
                    # Initialize RandomizedSearchCV
                    random_search = RandomizedSearchCV(
                        estimator=model,
                        param_distributions=param_dist,
                        n_iter=20,  # Number of parameter settings to sample
                        scoring='accuracy',  # Metric to evaluate
                        cv=5,  # Number of cross-validation folds
                        random_state=42,
                        n_jobs=-1,  # Use all available CPU cores
                    )
                elif model_type == 'SVM Classification':
                    model= SVC(random_state=random_state)
                    param_dist = {
                        'C': loguniform(1e-2, 1e2),  # Regularization parameter (log-uniform distribution)
                        'kernel': ['linear', 'rbf', 'poly'],  # Kernel type
                        'gamma': ['scale', 'auto'] + list(loguniform(1e-3, 1e1).rvs(5)),  # Kernel coefficient
                        'degree': randint(2, 5),  # Degree for polynomial kernel
                        'coef0': uniform(0.0, 1.0),  # Independent term in kernel
                        'shrinking': [True, False],  # Shrinking heuristic
                        'probability': [True, False],  # Probability estimates
                        'tol': loguniform(1e-4, 1e-2),  # Tolerance for stopping criteria
                        'class_weight': [None, 'balanced'],  # Class weights
                    }
                    # Initialize RandomizedSearchCV
                    random_search = RandomizedSearchCV(
                        estimator=model,
                        param_distributions=param_dist,
                        n_iter=5,  # Number of parameter settings to sample
                        scoring='accuracy',  # Metric to evaluate
                        cv=5,  # Number of cross-validation folds
                        random_state=42,
                        n_jobs=-1,  # Use all available CPU cores
                    )
                elif model_type =='Naive Bayes Classifier':
                    model= GaussianNB()
                    param_dist = {
                        'var_smoothing': loguniform(1e-9, 1e-5),  # Smoothing parameter
                    }
                    # Initialize RandomizedSearchCV
                    random_search = RandomizedSearchCV(
                        estimator=model,
                        param_distributions=param_dist,
                        n_iter=30,  # Number of parameter settings to sample
                        scoring='accuracy',  # Metric to evaluate
                        cv=5,  # Number of cross-validation folds
                        random_state=42,
                        n_jobs=-1,  # Use all available CPU cores
                    )
                elif model_type == 'KNeighborsClassifier':
                    model= KNeighborsClassifier()
                    param_dist = {
                        'n_neighbors': randint(3, 20),  # Number of neighbors
                        'weights': ['uniform', 'distance'],  # Weight function
                        'p': [1, 2],  # Power parameter for Minkowski metric
                        'metric': ['minkowski', 'euclidean', 'manhattan','chebyshev'],  # Algorithm for nearest neighbors
                    }
                    # Initialize RandomizedSearchCV
                    random_search = RandomizedSearchCV(
                        estimator=model,
                        param_distributions=param_dist,
                        n_iter=20,  # Number of parameter settings to sample
                        scoring='accuracy',  # Metric to evaluate (negative MSE for regression)
                        cv=5,  # Number of cross-validation folds
                        random_state=42,
                        n_jobs=-1,  # Use all available CPU cores
                    )
            else:
                if model_type == 'LinearRegression':
                    random_search= LinearRegression() #renamed from model for generality
                elif model_type == 'RandomForestRegressor':
                    model= RandomForestRegressor(random_state=random_state)
                    param_dist = {
                        'n_estimators': randint(50, 500),  # Number of trees
                        'max_depth': [None] + list(randint(10, 50).rvs(5)),  # Maximum depth
                        'min_samples_split': randint(2, 20),  # Minimum samples to split
                        'min_samples_leaf': randint(1, 10),  # Minimum samples at leaf
                        'max_features': ['auto', 'sqrt', 'log2', 0.5],  # Features to consider
                        'bootstrap': [True, False],  # Bootstrap sampling
                    }
                    # Initialize RandomizedSearchCV
                    random_search = RandomizedSearchCV(
                        estimator=model,
                        param_distributions=param_dist,
                        n_iter=30,  # Number of parameter settings to sample
                        scoring='neg_mean_squared_error',  # Metric to evaluate (negative MSE for regression)
                        cv=5,  # Number of cross-validation folds
                        random_state=42,
                        n_jobs=-1,  # Use all available CPU cores
                    )
                elif model_type == 'GradientBoostingRegressor':
                    model= GradientBoostingRegressor(random_state=random_state)
                    param_dist = {
                        'n_estimators': randint(50, 500),  # Number of boosting stages
                        'learning_rate': uniform(0.001, 0.2),  # Learning rate
                        'max_depth': randint(3, 15),  # Maximum depth of trees
                        'min_samples_split': randint(2, 20),  # Minimum samples to split
                        'min_samples_leaf': randint(1, 10),  # Minimum samples at leaf
                        'max_features': ['auto', 'sqrt', 'log2', 0.5],  # Features to consider
                        'subsample': uniform(0.5, 0.5),  # Fraction of samples for fitting
                        'loss': ['ls', 'lad', 'huber'],  # Loss function
                    }
                    # Initialize RandomizedSearchCV
                    random_search = RandomizedSearchCV(
                        estimator=model,
                        param_distributions=param_dist,
                        n_iter=20,  # Number of parameter settings to sample
                        scoring='neg_mean_squared_error',  # Metric to evaluate (negative MSE for regression)
                        cv=5,  # Number of cross-validation folds
                        random_state=42,
                        n_jobs=-1,  # Use all available CPU cores
                    )
                elif model_type == 'SVM Regression':
                    model= SVR()
                    param_dist = {
                        'kernel': ['linear', 'rbf', 'poly'],  # Kernel type
                        'C': loguniform(1e-2, 1e2),  # Regularization parameter (log-uniform distribution)
                        'gamma': ['scale', 'auto'] + list(loguniform(1e-3, 1e1).rvs(5)),  # Kernel coefficient
                        'epsilon': uniform(0.01, 1.0),  # Epsilon-tube
                        'degree': randint(2, 5),  # Degree for polynomial kernel
                        'coef0': uniform(0.0, 1.0),  # Independent term in kernel
                        'shrinking': [True, False],  # Shrinking heuristic
                        'tol': loguniform(1e-4, 1e-2),  # Tolerance for stopping criteria
                    }
                    random_search = RandomizedSearchCV(
                        estimator=model,
                        param_distributions=param_dist,
                        n_iter=5,  # Number of parameter settings to sample
                        scoring='neg_mean_squared_error',  # Metric to evaluate (negative MSE for regression)
                        cv=5,  # Number of cross-validation folds
                        random_state=42,
                        n_jobs=-1,  # Use all available CPU cores
                    )
                elif model_type == 'KNeighborsRegressor':
                    model= KNeighborsRegressor()
                    param_dist = {
                        'n_neighbors': randint(3, 15),  # Number of neighbors
                        'weights': ['uniform', 'distance'],  # Weight function
                        'p': [1, 2],  # Power parameter for Minkowski metric
                        'algorithm': ['auto', 'ball_tree', 'kd_tree'],  # Algorithm for nearest neighbors
                    }
                    # Initialize RandomizedSearchCV
                    random_search = RandomizedSearchCV(
                        estimator=model,
                        param_distributions=param_dist,
                        n_iter=20,  # Number of parameter settings to sample
                        scoring='neg_mean_squared_error',  # Metric to evaluate (negative MSE for regression)
                        cv=5,  # Number of cross-validation folds
                        random_state=42,
                        n_jobs=-1,  # Use all available CPU cores
                    )
                elif model_type == 'Ridge Regression':
                    model= Ridge()
                    param_dist = {
                        'alpha': uniform(0.1, 100),  # Regularization strength
                        'fit_intercept': [True, False],  # Whether to fit intercept
                        'solver': ['auto', 'svd', 'sag'],  # Solver for optimization
                    }
                    # Initialize RandomizedSearchCV
                    random_search = RandomizedSearchCV(
                        estimator=model,
                        param_distributions=param_dist,
                        n_iter=30,  # Number of parameter settings to sample
                        scoring='neg_mean_squared_error',  # Metric to evaluate (negative MSE for regression)
                        cv=5,  # Number of cross-validation folds
                        random_state=42,
                        n_jobs=-1,  # Use all available CPU cores
                    )
                elif model_type == 'Lasso Regression':
                    model= Lasso()
                    param_dist = {
                        'alpha': uniform(0.1, 100),  # Regularization strength
                        'fit_intercept': [True, False],  # Whether to fit intercept
                        'selection': ['cyclic', 'random'],  # Coefficient selection method
                    }
                    # Initialize RandomizedSearchCV
                    random_search = RandomizedSearchCV(
                        estimator=model,
                        param_distributions=param_dist,
                        n_iter=30,  # Number of parameter settings to sample
                        scoring='neg_mean_squared_error',  # Metric to evaluate (negative MSE for regression)
                        cv=5,  # Number of cross-validation folds
                        random_state=42,
                        n_jobs=-1,  # Use all available CPU cores
                    )
                elif model_type == 'ElasticNet Regression':
                    model= ElasticNet()
                    param_dist = {
                        'alpha': uniform(0.1, 100),  # Regularization strength
                        'l1_ratio': uniform(0.1, 1.0),  # Mixing parameter for L1/L2
                        'fit_intercept': [True, False],  # Whether to fit intercept
                        'selection': ['cyclic', 'random'],  # Coefficient selection method
                    }
                    # Initialize RandomizedSearchCV
                    random_search = RandomizedSearchCV(
                        estimator=model,
                        param_distributions=param_dist,
                        n_iter=30,  # Number of parameter settings to sample
                        scoring='neg_mean_squared_error',  # Metric to evaluate (negative MSE for regression)
                        cv=5,  # Number of cross-validation folds
                        random_state=42,
                        n_jobs=-1,  # Use all available CPU cores
                    )


            if df[target].dtype == 'object':
                random_search.fit(x_train, y_train)
                y_pred = random_search.best_estimator_.predict(x_test)
                score = accuracy_score(y_test, y_pred)
                cm = confusion_matrix(y_test, y_pred)
                cm_df= pd.DataFrame(cm,columns=df[target].unique(),index=df[target].unique())
                fig = px.imshow(cm_df, text_auto=True)
                report = classification_report(y_test, y_pred, target_names=df[target].unique())
                return html.Div([html.P(f"The metrics below are based on the test set of your data for {model_type}:"),
                                 html.P(f"Accuracy Score:{score}"), html.Pre(report),
                                 html.P('Confusion Matrix'),html.Pre(cm_df.to_string()),
                                 dcc.Graph(figure=fig), html.Pre(f"Best Parameters for {model_type}\n {random_search.best_params_}")])
            else:
                random_search.fit(x_train, y_train)
                if model_type == 'LinearRegression':
                    y_pred = random_search.predict(x_test)
                    mse = round(mean_squared_error(y_test, y_pred), 4)
                    rmse = np.sqrt(mse)
                    mae = mean_absolute_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)
                    adj_r2 = 1 - ((1 - r2) * (len(y_test) - 1) / (len(y_test) - len(features) - 1))
                    fig= go.Figure()
                    fig.add_trace(go.Scatter(y=y_test,mode='lines', name='Test Set'))
                    fig.add_trace(go.Scatter(y=y_pred, mode='lines', name='Predicted Set'))
                    fig.update_layout(
                        title=dict(
                            text=f'Graph of Test Set vs Prediction Set for {model_type} Model'))
                    return html.Div(
                        [html.P(f"The metrics below are based on the test set of your data for {model_type}:"),
                         html.P(f"Mean Squared Error:{mse}"), html.P(f"Root Mean Squared Error:{rmse}"),
                         html.P(f"Mean Absolute Error:{mae}"), html.P(f"R-Squared:{r2}"),
                         html.P(f"Adjusted R-Squared:{adj_r2}"), dcc.Graph(figure=fig)])
                else:
                    y_pred = random_search.best_estimator_.predict(x_test)
                    mse= round(mean_squared_error(y_test,y_pred),4)
                    rmse= np.sqrt(mse)
                    mae= mean_absolute_error(y_test,y_pred)
                    r2= r2_score(y_test, y_pred)
                    adj_r2= 1-((1-r2)*(len(y_test)-1)/(len(y_test)-len(features)-1))
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(y=y_test, mode='lines', name='Test Set'))
                    fig.add_trace(go.Scatter(y=y_pred, mode='lines', name='Predicted Set'))
                    fig.update_layout(
                        title=dict(
                            text=f'Graph of Test Set vs Prediction Set for {model_type} Model'))
                    return html.Div([html.P(f"The metrics below are based on the test set of your data for {model_type}:"),
                                     html.P(f"Mean Squared Error:{mse}"),html.P(f"Root Mean Squared Error:{rmse}"),
                                     html.P(f"Mean Absolute Error:{mae}"),html.P(f"R-Squared:{r2}"),
                                     html.P(f"Adjusted R-Squared:{adj_r2}"),
                                     html.Pre(f"Best Parameters for {model_type}\n {random_search.best_params_}"),
                                     dcc.Graph(figure=fig)])

    except:
        return ""
#======================================================================================

@my_app.callback(Output('download-data-cleaned','data'),
                 [Input('radio','value'),
                  Input('store1','data'),
                  Input('btn_csv','n_clicks')], prevent_initial_call=True)
def out(value, data, n_clicks):
    if value == 'Cleaned Data' and n_clicks:
        df = pd.read_json(data, orient='table')
        return dcc.send_data_frame(df.to_csv, 'cleaned_data.csv')



if __name__ == '__main__':
    my_app.run_server(
        debug = True,
        host= '0.0.0.0',
        port= 8080
    )