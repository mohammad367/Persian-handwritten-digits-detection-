import joblib
import sklearn
model = joblib.load('models/knn_model.pkl')

def detector(image):
    return  model.predict(image)[0]