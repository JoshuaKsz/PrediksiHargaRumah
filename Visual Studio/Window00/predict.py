import sys
import ast
import joblib
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def main(array_string):
    try:
        array = ast.literal_eval(array_string)
        
        if not isinstance(array, list):
            raise ValueError("Provided input is not a list")
        
        print(f"The array passed is: {array}")
       
        X_convert = pd.DataFrame({
            "jumlah kamar tidur": [int(array[0])],
            "jumlah kamar mandi": [int(array[1])],
            "luas tanah (m2)": [int(array[2])],
            "luas bangunan (m2)": [int(array[3])],
            "carport (mobil)": [int(array[4])],
            "pasokan listrik (watt)": [int(array[5])],
            "Kabupaten/Kota": [array[6]],
            'kecamatan' : [array[7]],
            "keamanan (ada/tidak)": [array[8]],
            "taman (ada/tidak)": [array[9]],
            "jarak dengan rumah sakit terdekat (km)": [float(array[10])],
            "jarak dengan sekolah terdekat (km)": [float(array[11])],
            "jarak dengan tol terdekat (km)": [float(array[12])]
        })
        print(X_convert)
        
        with open('model.txt', 'r') as file:
            nama_model = file.read()

        print("nama model: ",nama_model)
        model = joblib.load(nama_model)
        
        
        predictions = model.predict(X_convert)
        readable_predictions = [f"{pred:,.0f}" for pred in predictions]
        readable_predictions[0] = readable_predictions[0].replace(",", "")
        with open('hasil.txt', 'w') as file:
            file.write(str(readable_predictions[0]))
        
        print("Prediction written to 'hasil.txt'")
    
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python test.py <array>")
        sys.exit(1)
    
    array_string = sys.argv[1]
    main(array_string)
    
    
