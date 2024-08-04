from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.cluster import AffinityPropagation 
from sklearn.metrics.pairwise import pairwise_distances_argmin_min
from sklearn.preprocessing import StandardScaler 
import pickle 
app = Flask(__name__)      
def canopy_feature_selection(X, threshold):
    canopy_centers = []       
    canopy_points = []        
    for i, point in enumerate(X):     
        if len(canopy_centers) == 0:
            canopy_centers.append(point)
        else:
            distances = [((c - point) ** 2).sum() for c in canopy_centers]
            min_distance = min(distances)
            closest_center = distances.index(min_distance)
            if min_distance < threshold:
                canopy_points[closest_center].append(point)
            else:
                canopy_centers.append(point)
                canopy_points.append([point])
    return canopy_centers

def make_prediction(url, dt_model, rf_model, xgb_model):
    # Placeholder code for feature extraction (replace with your actual feature extraction code)
    # For example, you can extract features like URL length, presence of certain keywords, etc.
    # For demonstration, let's assume we're extracting a simple feature: URL length
    url_length = len(url)
    
    # Placeholder code for making prediction using the models (replace with your actual prediction code)
    # For example, you can use each model to make predictions based on the extracted features
    dt_prediction = dt_model.predict([[url_length]])[0]
    rf_prediction = rf_model.predict([[url_length]])[0]
    xgb_prediction = xgb_model.predict([[url_length]])[0]
    
    # Take a majority vote or apply your own logic to combine the predictions
    # For demonstration, let's assume we're taking a majority vote
    if dt_prediction + rf_prediction + xgb_prediction >= 2:
        prediction = "Phishing"
    else:
        prediction = "Not Phishing"
    
    return prediction


dataset = pd.read_csv('phishing.csv')
dataset.dropna(inplace=True)
X = dataset.drop(columns=['class'])
y = dataset['class']

X_canopy = canopy_feature_selection(X.values, threshold=0.5)
X_canopy_df = pd.DataFrame(X_canopy, columns=X.columns)

X_train, X_test, y_train, y_test = train_test_split(X_canopy_df, y, test_size=0.3, random_state=42)

dt_model = DecisionTreeClassifier()
rf_model = RandomForestClassifier()
xgb_model = XGBClassifier()

voting_model = VotingClassifier([('dt', dt_model), ('rf', rf_model), ('xgb', xgb_model)], voting='hard')

param_grid = {
    'dt__max_depth': [5, 10, 15],
    'rf__n_estimators': [50, 100, 200],
    'xgb__n_estimators': [50, 100, 200]
}

grid_search = GridSearchCV(voting_model, param_grid, cv=5)
grid_search.fit(X_train, y_train)


y_pred = grid_search.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)


with open('dt_model.pkl', 'wb') as f:
    pickle.dump(dt_model, f)

with open('rf_model.pkl', 'wb') as f:
    pickle.dump(rf_model, f)

with open('xgb_model.pkl', 'wb') as f:
    pickle.dump(xgb_model, f)



@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    url = request.json['url']
    with open('dt_model.pkl', 'rb') as f:
        dt_model = pickle.load(f)

    with open('rf_model.pkl', 'rb') as f:
        rf_model = pickle.load(f)

    with open('xgb_model.pkl', 'rb') as f:
        xgb_model = pickle.load(f)

    prediction = make_prediction(url, dt_model, rf_model, xgb_model)
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)