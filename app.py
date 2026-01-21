from flask import Flask, request, render_template
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

app = Flask(__name__)

df = pd.read_csv('cancer.csv')

selected_features = ['Air Pollution', 'Genetic Risk', 'Obesity', 'Balanced Diet', 'OccuPational Hazards', 'Coughing of Blood']
x = df[selected_features]
y = df['Level']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(x_train, y_train)

y_pred = rf_clf.predict(x_test)

acc = accuracy_score(y_test, y_pred)
class_rep = classification_report(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print(f'Accuracy: {acc:.2f}')
print('\nClassification Report:\n', class_rep)
print('\nConfusion Matrix:\n', cm)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    air_pol = float(request.form['air_pollution'])
    gen_risk = float(request.form['genetic_risk'])
    obesity = float(request.form['obesity'])
    bal_diet = float(request.form['balanced_diet'])
    occ_haz = float(request.form['occupational_hazards'])
    cough_blood = float(request.form['coughing_of_blood'])

    input_data = [[air_pol, gen_risk, obesity, bal_diet, occ_haz, cough_blood]]

    prediction = rf_clf.predict(input_data)
    prediction_prob = rf_clf.predict_proba(input_data)

    predicted_class = prediction[0]
    predicted_prob = prediction_prob[0]

    result = {
        'predicted_class': predicted_class,
        'predicted_probability': dict(zip(rf_clf.classes_, predicted_prob))
        }
    
    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)

