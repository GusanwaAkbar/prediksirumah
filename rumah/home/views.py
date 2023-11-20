from django.shortcuts import render
from home.models import Table_rumah
from folium import IFrame



#import machine learning setup
import pandas as pd
import sklearn
import numpy as np


#import joblib
from joblib import load

#importing GIS
import folium

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error



def main_map(request):

    selected_columns = [
        'Titik_koordinat',
        'Nama_debitur',
        'Nama_Pemberi_Tugas',
        'Kota_Kabupaten',
        'LT',
        'LB',
        'Tujuan',
        'Obyek',
        'Fee_total',
        'dpp',
        'ppn',
        'Indikasi_nilai',
        'Jumlah_kamar',
        'Jumlah_kamarmandi',
        'Jumlah_lantai',
        'Pusat_kota',
    ]

    queryset = Table_rumah.objects.all()

    # Convert the queryset to a pandas DataFrame
    df = pd.DataFrame(list(queryset.values()))
    #df selected columns
    df_print = df[selected_columns]
    
    #make the map
    map_html = make_map(df_print)

    # Display the DataFrame
    print(df.head())

    context = {'map_html': map_html,
               'df': df_print,
               }




    return render(request, 'index.html', context)



def make_map(df):
    asset_map = folium.Map(location=[-7.9797, 112.6304], zoom_start=13, width="100%", height="100%", control_scale=True)

    # Iterate over rows in your DataFrame
    for index, row in df.iterrows():
        # Extract latitude and longitude from 'Titik_koordinat' column
        str_koor = row['Titik_koordinat']
        latitude = float(str_koor[0:8])
        longitude = float(str_koor[11:23])

        # Create an HTML table string using data from the DataFrame
        table_html = "<table>"
        for key, value in row.items():
            table_html += f"<tr><td>{key}</td><td>{value}</td></tr>"
        table_html += "</table>"

        # Create an IFrame with the HTML content
        popup_html = IFrame(html=table_html, width=300, height=200)

        # Add a marker to the map with the table-style popup
        folium.Marker(
            location=[latitude, longitude],
            popup=folium.Popup(popup_html, max_width=300),
            icon=folium.Icon(color='red')
        ).add_to(asset_map)

    map_html = asset_map._repr_html_()

    return map_html


def make_predict(df,model_name,split):

    df["Kota/Kabupaten"] = df.Kota_Kabupaten

    print("**********************")
    
    print(df["Kota/Kabupaten"])

    categorical_columns = ["Kota/Kabupaten", "Tujuan", "Obyek", "Pusat_kota"]

    features = ["Kota/Kabupaten", 
                "LT", 
                "LB",
                "Tujuan",
                "Obyek",
                "Indikasi_nilai",
                "Jumlah_kamar",
                "Jumlah_kamarmandi",
                "Jumlah_lantai",
                "Pusat_kota",
                ]

    data_features = df[features]
    data_features

    # Use one-hot encoding for categorical columns
    X = pd.get_dummies(data_features, columns=categorical_columns)

    # Change null with means value
    X.Indikasi_nilai = df.Indikasi_nilai.replace(0, X.Indikasi_nilai.mean())

    #Define Target Y
    y = df.Kesimpulan_nilai

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split, random_state=42)

    #INT INPUT
    X = X.astype(int)
    y = y.astype(int)

    loaded_model = load('../notebook/'+ model_name +'.joblib')
    print("===========================")
    print(X_test)

    predictions = loaded_model.predict(X_test)
    
    
    return predictions, y_test


def MAPE(data_predict, y_test):

    arr_error = abs(data_predict-y_test)
    mean_error = np.mean(arr_error)

    

    return data_predict, y_test, arr_error, mean_error




# Create your views here.
def home(request):

    selected_columns = [
        'Titik_koordinat',
        'Nama_debitur',
        'Nama_Pemberi_Tugas',
        'Kota_Kabupaten',
        'LT',
        'LB',
        'Tujuan',
        'Obyek',
        'Fee_total',
        'dpp',
        'ppn',
        'Indikasi_nilai',
        'Jumlah_kamar',
        'Jumlah_kamarmandi',
        'Jumlah_lantai',
        'Pusat_kota',
    ]

    # Query all objects from the Table_rumah model
    queryset = Table_rumah.objects.all()

    # Convert the queryset to a pandas DataFrame
    df = pd.DataFrame(list(queryset.values()))
    #df selected columns
    df_print = df[selected_columns]
    
    #make the map
    map_html = make_map(df_print)

    # Display the DataFrame
    print(df.head())




    #Predict
    data_predict, data_asli = make_predict(df, "Model2", 0.5)

    print("/////////////////////////")
    print(data_asli)
    
    arr_asli = []

    for x in data_asli:
        arr_asli.append(float(x))

    print("@@@@@@@@@@@@@@@@@@@@@@@@")
    print(data_predict)

    

    #MAPE
    predict,asli , arr_error, mean_error = MAPE(data_predict, arr_asli)


    data = {
    'Predict': predict,
    'Asli': asli,
    'Error': arr_error
    }

    # Create a DataFrame from the dictionary
    df_y = pd.DataFrame(data)

    df_merged = pd.merge(df, df_y, left_index=True, right_index=True)

    df_merged["asli"] = asli
    df_merged["predict"] = predict
    df_merged["error"] = arr_error
    



    # Convert the Folium asset_map to HTML
    df_list = df_merged.to_dict(orient='records')

    

    # Pass the map HTML to the template
    context = {'map_html': map_html,
               'df': df_list,
               'predict': predict,
               'asli': asli,
               'error': arr_error,
               'mean_error': mean_error,
               }






    






    
    return render(request, 'index3.html', context)