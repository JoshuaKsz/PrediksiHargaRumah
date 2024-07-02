import pandas as pd
data = pd.read_csv("Data Harga Rumah di Jakarta - All Data.csv")
# jika memakai jupyter ubah comment yang atas dan uncomment yang bawah
# data = pd.read_csv("Data Harga Rumah di Jakarta - All Data.csv")
len_baris = len(data)
print(data.head())

"""# Libraries"""

import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
import lightgbm as lgb
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR

"""# Preprocess Data"""

# hapus kolom data yang tidak dipakai
columns_to_drop = ['No', 'NRP', 'Nama', 'kelurahan', 'Link Data Rumah']
data.drop(columns=[col for col in columns_to_drop if col in data.columns], inplace=True)

# hapus data yang paling kanan Unnamed: 19 * contoh data
data.drop(columns=['Unnamed: 19'], inplace=True)

data.fillna(method='ffill', inplace=True)

# melakukan replace pada ',' karena komputer hanya membaca '.' sebagai decimal
# dikarenakan Amerika Serikat yang menemukan komputer terlebih dahulu yang memakai pemisah desimal '.'
for col in ['jarak dengan rumah sakit terdekat (km)',
            'jarak dengan sekolah terdekat (km)', 'jarak dengan tol terdekat (km)']:
    data[col] = data[col].str.replace(',', '.', regex=True).astype(float)

# dengan berpikir dengan pattern recognition saya melihat ada yang memakai ',' dan '.' dan juga ada yang memakai .00 dan "Rp " dan "Rp"
# dengan demikian saya menghapus 3 karakter jika memiliki dibelakanya ".00" kemudian replace ',' dan '.' menjadi '' sehingga string dapat
# menjadi satu menjadi intger, dan juga membuang 'Rp' dan 'Rp ' menjadi ''
def clean_and_convert(index, amount_str):
    # menghapus RP '.' ',' dengan membuang belakang .00 dan menghitung jumlah 0 nya

    text = (amount_str)

    if text.endswith(".00"):
        text = text[:-3]
    else:
        pass
    text = text.replace('.', '').replace(',', '')
    text = text.replace('Rp', '').replace('Rp ', '')
    text = text.replace(' ', '')
    return int(text)

data['harga rumah'] = data.apply(lambda row: clean_and_convert(row.name, row['harga rumah']), axis=1)

# Melakukkan Pengecekan data yang ingin di one hot
def cek_data(colom):
    uniq = []
    for value in data[colom]:
        if value not in uniq:
            uniq.append(value)
    print("Unique values", colom)
    print(uniq)

cek_data("keamanan (ada/tidak)")

cek_data("taman (ada/tidak)")

cek_data("Kabupaten/Kota")

# ubah menjadi semua data menjadi huruf kecil
data['keamanan (ada/tidak)'] = data['keamanan (ada/tidak)'].str.lower()
data['taman (ada/tidak)'] = data['taman (ada/tidak)'].str.lower()

# Dengan memakai pattern recognition saya melihat bahwa semua data tidak serupa jadi saya meilhat persamaannya
# ternyata yang string artinya 'tidak' seperti 'tdk', 'tdak', dll. memiliki awalan huruf 't' dan 'ada' tidak memiliki awalan 't'

# sebelum itu hapus terlebih dahulu data yang kosong
data = data[data['keamanan (ada/tidak)'].notna() & (data['keamanan (ada/tidak)'] != '')]
data = data[data['taman (ada/tidak)'].notna() & (data['taman (ada/tidak)'] != '')]

def map_tidak(kata):
    if kata[0].lower() == 't':
        return 'tidak'
    else:
        return 'ada'

# mengubah semuannya menjadi tidak dan ada
data['keamanan (ada/tidak)'] = data['keamanan (ada/tidak)'].apply(map_tidak)
data['taman (ada/tidak)'] = data['taman (ada/tidak)'].apply(map_tidak)

# mengecek data lainnya
cek_data("jumlah kamar tidur")
cek_data("jumlah kamar mandi")
cek_data("luas tanah (m2)")
cek_data("luas bangunan (m2)")
cek_data("carport (mobil)")
cek_data("pasokan listrik (watt)")
cek_data("jarak dengan rumah sakit terdekat (km)")
cek_data("jarak dengan sekolah terdekat (km)")
cek_data("jarak dengan tol terdekat (km)")
cek_data("kecamatan")

# menghapus data yang tidak ada isinya '' / null
data = data[data['jumlah kamar tidur'].notna() & (data['jumlah kamar tidur'] != '')]
data = data[data['jumlah kamar mandi'].notna() & (data['jumlah kamar mandi'] != '')]
data = data[data['luas tanah (m2)'].notna() & (data['luas tanah (m2)'] != '')]
data = data[data['luas bangunan (m2)'].notna() & (data['luas bangunan (m2)'] != '')]
data = data[data['pasokan listrik (watt)'].notna() & (data['pasokan listrik (watt)'] != '')]
data = data[data['jarak dengan rumah sakit terdekat (km)'].notna() & (data['jarak dengan rumah sakit terdekat (km)'] != '')]
data = data[data['jarak dengan sekolah terdekat (km)'].notna() & (data['jarak dengan sekolah terdekat (km)'] != '')]
data = data[data['jarak dengan tol terdekat (km)'].notna() & (data['jarak dengan tol terdekat (km)'] != '')]
data = data[data['carport (mobil)'].notna() & (data['carport (mobil)'] != '')]
data = data[data['kecamatan'].notna() & (data['kecamatan'] != '')]

# mustahil untuk rumah tidak memiliki luas tanah (m2), luas bangunan (m2)
data = data[(data['luas tanah (m2)'] != '0') & (data['luas tanah (m2)'] != 0)]
data = data[(data['luas bangunan (m2)'] != '0') & (data['luas tanah (m2)'] != 0)]

# Sangat Jarang juga rumah yang tidak memiliki pasokan listrik
data = data[(data['pasokan listrik (watt)'] != '0') & (data['pasokan listrik (watt)'] != 0) & (data['pasokan listrik (watt)'] != '-') & (data['pasokan listrik (watt)'] != '--')]

# terdapat ',' didata jadi dihilangkan untuk dapat di convert ke integer
data['pasokan listrik (watt)'] = data['pasokan listrik (watt)'].astype(str).str.replace(',', '')

# '-' mungkin isisnya 0 dan
data['jumlah kamar tidur'] = data['jumlah kamar tidur'].replace('-', '0')
data['jumlah kamar mandi'] = data['jumlah kamar mandi'].replace('-', '0')
data['carport (mobil)'] = data['carport (mobil)'].replace('-', '0')

# Ubah terlebih dahulu tipe data jika data berupa numerik
data['jumlah kamar tidur'] = data['jumlah kamar tidur'].astype(int)
data['jumlah kamar mandi'] = data['jumlah kamar mandi'].astype(int)
data['luas tanah (m2)'] = data['luas tanah (m2)'].astype(int)
data['luas bangunan (m2)'] = data['luas bangunan (m2)'].astype(int)
data['carport (mobil)'] = data['carport (mobil)'].astype(int)
data['pasokan listrik (watt)'] = data['pasokan listrik (watt)'].astype(int)

# huruf pertama pada kecamatan diganti menjadi besar
data['kecamatan'] = data['kecamatan'].apply(lambda name: ' '.join([part.capitalize() for part in name.split()]))

# Entah Mengapa ada 2 penulisan Pasar Rebo: yaitu Ps. Rebo dan Pasar Rebo
# Jadi saya ubah Ps. Rebo menjadi Pasar Rebo
data['kecamatan'] = data['kecamatan'].replace('Ps. Rebo', 'Pasar Rebo')

# cek ulang data
cek_data("keamanan (ada/tidak)")
cek_data("taman (ada/tidak)")
cek_data("Kabupaten/Kota")
cek_data("jumlah kamar tidur")
cek_data("jumlah kamar mandi")
cek_data("luas tanah (m2)")
cek_data("luas bangunan (m2)")
cek_data("carport (mobil)")
cek_data("pasokan listrik (watt)")
cek_data("jarak dengan rumah sakit terdekat (km)")
cek_data("jarak dengan sekolah terdekat (km)")
cek_data("jarak dengan tol terdekat (km)")
cek_data("kecamatan")

# kalkulasi Jumlah data awal dan akhir
print('perhitungan jumlah data perbaris yang sudah di Data preprocessing atau data preparation')
ak = len(data)
print('data akhir:', ak)
print('data awal:', len_baris)
print('Total penghapusan', len_baris - ak)

# penghapusan data yang mustahi;
data = data[data["jarak dengan rumah sakit terdekat (km)"] <= 50]
data = data[data["jarak dengan sekolah terdekat (km)"] <= 50]
data = data[data["jarak dengan tol terdekat (km)"] <= 50]

print("data yang dihapus: ")
print('data akhir:', len(data))
print('data awal:', ak)
print('Total penghapusan', ak - len(data))

y = data['harga rumah']
X = data.drop(columns=['harga rumah'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X

y

print(data.columns)

"""# Models dan Hyperparameter Tuning"""

from sklearn.preprocessing import PolynomialFeatures

categorical_features = ['Kabupaten/Kota', 'kecamatan', 'keamanan (ada/tidak)', 'taman (ada/tidak)']
numerical_features = data.columns.difference(categorical_features + ['harga rumah']).tolist()


poly = PolynomialFeatures(degree=2, include_bias=False)
numerical_transformer = Pipeline(steps=[
    ('scaler', StandardScaler()),
    ('poly', poly)
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])



models = {
    'LinearRegression': LinearRegression(),
    'DecisionTree': DecisionTreeRegressor(),
    'RandomForest': RandomForestRegressor(),
    'XGBRegressor': XGBRegressor(),
    'GradientBoosting': GradientBoostingRegressor(),
    'LightGBM': lgb.LGBMRegressor(),
    'SVR': SVR()
}

# Model nya underfitting jadi parameternya saya ganti
param_grid = {
    'RandomForest': {
        'model__n_estimators': [50, 100, 200, 500],
        'model__max_depth': [10, 20, 30, None],
        'model__min_samples_split': [2, 5, 10],
        'model__min_samples_leaf': [1, 2, 4]
    },
    'XGBRegressor': {
        'model__n_estimators': [50, 100, 200, 500],
        'model__learning_rate': [0.01, 0.1, 0.2, 0.3],
        'model__max_depth': [3, 6, 9]
    },
    'GradientBoosting': {
        'model__n_estimators': [50, 100, 200, 500],
        'model__learning_rate': [0.01, 0.1, 0.2, 0.3],
        'model__max_depth': [3, 4, 5]
    },
    'LightGBM': {
        'model__n_estimators': [50, 100, 200, 500],
        'model__learning_rate': [0.01, 0.1, 0.2, 0.3],
        'model__num_leaves': [31, 50, 100]
    },
    'SVR': {
        'model__C': [0.1, 1, 10, 100],
        'model__epsilon': [0.01, 0.1, 0.2],
        'model__kernel': ['linear', 'rbf']
    }
}

best_models = {}
for model_name in models:
    model = Pipeline(steps=[('preprocessor', preprocessor),
                            ('model', models[model_name])])

    if model_name in param_grid:
        search = GridSearchCV(model, param_grid[model_name], cv=5, scoring='r2', n_jobs=-1)
    else:
        search = model

    search.fit(X_train, y_train)
    best_models[model_name] = search.best_estimator_ if model_name in param_grid else model

"""# Evaluasi Data"""

results = {}
for model_name in best_models:
    model = best_models[model_name]
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    results[model_name] = {'R²': r2, 'MAE': mae, 'RMSE': rmse}

results_df = pd.DataFrame(results).T
print(results_df)

"""# Best Model dan Hasil"""

best_model_name = results_df['R²'].idxmax()
best_model = best_models[best_model_name]

print(f"The best model is {best_model_name} with an R² score of {results_df.loc[best_model_name, 'R²']:.2f}")

y_pred = best_model.predict(X_test)

comparison_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(comparison_df.head())

from sklearn.model_selection import cross_val_score

cv = KFold(n_splits=10, random_state=42, shuffle=True)
cv_results = cross_val_score(best_model, X, y, cv=cv, scoring='r2')

print(f"Cross-validated R² score: {cv_results.mean():.2f} ± {cv_results.std():.2f}")


"""# Model Terbaik = Random Forest"""

skor = best_model.score(X_test, y_test)
print(skor)

"""SIMPAN BEST MODEL"""

import joblib
model_filename = f"{best_model_name}_model_normal_{round(skor, 2)}.joblib"
joblib.dump(best_model, model_filename)
print(f"Model saved as {model_filename}")


"""Coba Lagi tetapi dengan menghapus outlier"""

def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1

    # Define the outlier boundaries
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Filter the DataFrame to remove outliers
    df_no_outliers = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

    # Calculate number of outliers removed
    outliers_removed = len(df) - len(df_no_outliers)

    print(outliers_removed)
    return df_no_outliers

data = remove_outliers(data, 'harga rumah')
data = remove_outliers(data, 'jarak dengan rumah sakit terdekat (km)')
data = remove_outliers(data, 'jarak dengan sekolah terdekat (km)')

print("outliers yang dihapus: ")
print('Total data akhir:', len(data))

y = data['harga rumah']
X = data.drop(columns=['harga rumah'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

best_models = {}
for model_name in models:
    model = Pipeline(steps=[('preprocessor', preprocessor),
                            ('model', models[model_name])])

    if model_name in param_grid:
        search = GridSearchCV(model, param_grid[model_name], cv=5, scoring='r2', n_jobs=-1)
    else:
        search = model

    search.fit(X_train, y_train)
    best_models[model_name] = search.best_estimator_ if model_name in param_grid else model

results = {}
for model_name in best_models:
    model = best_models[model_name]
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    results[model_name] = {'R²': r2, 'MAE': mae, 'RMSE': rmse}

results_df = pd.DataFrame(results).T
print(results_df)

best_model_name = results_df['R²'].idxmax()
best_model = best_models[best_model_name]

print(f"The best model is {best_model_name} with an R² score of {results_df.loc[best_model_name, 'R²']:.2f}")

y_pred = best_model.predict(X_test)

comparison_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(comparison_df.head())

cv = KFold(n_splits=10, random_state=42, shuffle=True)
cv_results = cross_val_score(best_model, X, y, cv=cv, scoring='r2')

print(f"Cross-validated R² score: {cv_results.mean():.2f} ± {cv_results.std():.2f}")
skor = best_model.score(X_test, y_test)
print("model terbaik setelah hapus data yang mustahil: ",skor)

model_filename = f"{best_model_name}_model_IQR_{round(skor, 2)}.joblib"
joblib.dump(best_model, model_filename)
print(f"Model saved as {model_filename}")