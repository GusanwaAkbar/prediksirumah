from django.shortcuts import render
from home.models import Table_rumah
from home.models import Simulasi
from folium import IFrame
from django.http import JsonResponse

from sklearn.preprocessing import LabelEncoder


from django.views.decorators.csrf import csrf_exempt  # Add this line



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

from sklearn import tree
import tempfile



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

    tree_texts = []
    for i, tree_in_forest in enumerate(loaded_model.estimators_):
        tree_text = tree.export_text(tree_in_forest, feature_names=X_train.columns.tolist())
        tree_texts.append(tree_text)

    # Combine text representations into one
    combined_text = "\n".join(tree_texts)

    # Make predictions on the test set
    predictions = loaded_model.predict(X_test)
    
    
    return predictions, y_test, combined_text


def predict_single_row(data, model_name):
    categorical_columns = ["Kota/Kabupaten", "Tujuan", "Obyek", "Pusat_kota"]

    # Convert the input data to a DataFrame
    new_data = pd.DataFrame([data])

    queryset = Table_rumah.objects.all()

    # Load the label encoder
    df = pd.DataFrame(list(queryset.values()))
    df["Kota/Kabupaten"] = df.Kota_Kabupaten
    

    # Use label encoding for categorical columns
    for column in categorical_columns:
    
        le = LabelEncoder()
        le.fit(df[column])
        new_data[column] = le.transform(new_data[column])

    # Load the model
    loaded_model = load('/home/gusanwa/AA_Programming/huda/rumah/notebook/' + model_name + '.joblib')

    # Make prediction
    prediction = loaded_model.predict(new_data)

    return prediction[0]  # Assuming the model returns an array, extract the first element




def MAPE(data_predict, y_test):

    arr_error = abs(data_predict-y_test)
    mean_error = np.mean(arr_error)

    error2 = np.abs((data_predict - y_test) / y_test) * 100
    mape = np.mean(error2)

    

    return data_predict, y_test, arr_error, mean_error, mape




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



# Create your views here.
def run1(request):

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
    data_predict, data_asli, tree = make_predict(df, "Model1", 0.1)

    print("/////////////////////////")
    print(data_asli)
    
    arr_asli = []

    for x in data_asli:
        arr_asli.append(float(x))

    print("@@@@@@@@@@@@@@@@@@@@@@@@")
    print(data_predict)

    

    #MAPE
    predict,asli , arr_error, mean_error, mape = MAPE(data_predict, arr_asli)


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

    total_error = np.sum(arr_error)
    n_data = len(arr_error)


    



    # Convert the Folium asset_map to HTML
    df_list = df_merged.to_dict(orient='records')

    total_error = np.sum(arr_error)
    n_data = len(arr_error)
    

    # Pass the map HTML to the template
    context = {'map_html': map_html,
               'df': df_list,
               'predict': predict,
               'asli': asli,
               'error': arr_error,
               
               'total_error' : total_error,
               'n_data': n_data,
               'mean_error': mean_error,
               'mape': mape,

               'tree': tree,

               }




    
    return render(request, 'index3.html', context)


def run2(request):

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
    data_predict, data_asli , tree = make_predict(df, "Model2", 0.2)

    print("/////////////////////////")
    print(data_asli)
    
    arr_asli = []

    for x in data_asli:
        arr_asli.append(float(x))

    print("@@@@@@@@@@@@@@@@@@@@@@@@")
    print(data_predict)

    

    #MAPE
    predict,asli , arr_error, mean_error, mape = MAPE(data_predict, arr_asli)


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
    

    df_list = df_merged.to_dict(orient='records')

    # Convert the Folium asset_map to HTML
    total_error = np.sum(arr_error)
    n_data = len(arr_error)
    

    # Pass the map HTML to the template
    context = {'map_html': map_html,
               'df': df_list,
               'predict': predict,
               'asli': asli,
               'error': arr_error,
               
               'total_error' : total_error,
               'n_data': n_data,
               'mean_error': mean_error,
               'mape':mape,
               'tree':tree,

               }




    
    return render(request, 'index3.html', context)


def run3(request):

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
    data_predict, data_asli, tree = make_predict(df, "Model3", 0.3)

    print("/////////////////////////")
    print(data_asli)
    
    arr_asli = []

    for x in data_asli:
        arr_asli.append(float(x))

    print("@@@@@@@@@@@@@@@@@@@@@@@@")
    print(data_predict)

    

    #MAPE
    predict,asli , arr_error, mean_error, mape = MAPE(data_predict, arr_asli)


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
    

    df_list = df_merged.to_dict(orient='records')

    # Convert the Folium asset_map to HTML
    total_error = np.sum(arr_error)
    n_data = len(arr_error)
    

    # Pass the map HTML to the template
    context = {'map_html': map_html,
               'df': df_list,
               'predict': predict,
               'asli': asli,
               'error': arr_error,
               
               'total_error' : total_error,
               'n_data': n_data,
               'mean_error': mean_error,

               'mape' : mape,
               'tree' : tree,

               }



    
    return render(request, 'index3.html', context)

def run4(request):

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
    data_predict, data_asli, tree = make_predict(df, "Model4", 0.4)

    print("/////////////////////////")
    print(data_asli)
    
    arr_asli = []

    for x in data_asli:
        arr_asli.append(float(x))

    print("@@@@@@@@@@@@@@@@@@@@@@@@")
    print(data_predict)

    

    #MAPE
    predict,asli , arr_error, mean_error, mape = MAPE(data_predict, arr_asli)


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

    

    total_error = np.sum(arr_error)
    n_data = len(arr_error)
    

    # Pass the map HTML to the template
    context = {'map_html': map_html,
               'df': df_list,
               'predict': predict,
               'asli': asli,
               'error': arr_error,
               
               'total_error' : total_error,
               'n_data': n_data,
               'mean_error': mean_error,
               'mape' : mape,

               'tree' : tree,

               }




    
    return render(request, 'index3.html', context)



def run5(request):

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
    data_predict, data_asli , tree = make_predict(df, "Model5", 0.5)

    print("/////////////////////////")
    print(data_asli)
    
    arr_asli = []

    for x in data_asli:
        arr_asli.append(float(x))

    print("@@@@@@@@@@@@@@@@@@@@@@@@")
    print(data_predict)

    

    #MAPE
    predict,asli , arr_error, mean_error, mape = MAPE(data_predict, arr_asli)


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
    

    total_error = np.sum(arr_error)
    n_data = len(arr_error)

    # Convert the Folium asset_map to HTML
    df_list = df_merged.to_dict(orient='records')

    

    # Pass the map HTML to the template
    context = {'map_html': map_html,
               'df': df_list,
               'predict': predict,
               'asli': asli,
               'error': arr_error,
               
               'total_error' : total_error,
               'n_data': n_data,
               'mean_error': mean_error,
               'mape' : mape,
               'tree': tree,
               }




    
    return render(request, 'index3.html', context)


def simulasi_list(request):
    simulasi_data = Simulasi.objects.all()
    return render(request, 'predict.html', {'simulasi_data': simulasi_data})

@csrf_exempt
def add_simulasi(request):
    if request.method == 'POST':
        nama = request.POST.get('Nama')
        titik_koordinat = request.POST.get('Titik_koordinat')
        kota_kabupaten = request.POST.get('Kota_Kabupaten')
        lt = request.POST.get('LT')
        lb = request.POST.get('LB')
        tujuan = request.POST.get('Tujuan')
        obyek = request.POST.get('Obyek')
        indikasi_nilai = request.POST.get('Indikasi_nilai')
        jumlah_kamar = request.POST.get('Jumlah_kamar')
        jumlah_kamarmandi = request.POST.get('Jumlah_kamarmandi')
        jumlah_lantai = request.POST.get('Jumlah_lantai')
        pusat_kota = request.POST.get('Pusat_kota')

        data_input = {
            "Kota/Kabupaten": kota_kabupaten,
            "LT": int(lt),  # Convert to int
            "LB": int(lb),  # Convert to int
            "Tujuan": tujuan,
            "Obyek": obyek,
            "Indikasi_nilai": int(indikasi_nilai),  # Convert to float
            "Jumlah_kamar": int(jumlah_kamar),  # Convert to int
            "Jumlah_kamarmandi": int(jumlah_kamarmandi),  # Convert to int
            "Jumlah_lantai": int(jumlah_lantai),  # Convert to int
            "Pusat_kota": pusat_kota,
        }

        # Get your model name (replace 'your_model_name' with your actual model name)
        model_name = 'Model1'

        # Use the predict_single_row function
        prediksi = predict_single_row(data_input, model_name)

        

        Simulasi.objects.create(
            Nama=nama,
            Titik_koordinat=titik_koordinat,
            Kota_Kabupaten=kota_kabupaten,
            LT=lt,
            LB=lb,
            Tujuan=tujuan,
            Obyek=obyek,
            Indikasi_nilai=indikasi_nilai,
            Jumlah_kamar=jumlah_kamar,
            Jumlah_kamarmandi=jumlah_kamarmandi,
            Jumlah_lantai=jumlah_lantai,
            Pusat_kota=pusat_kota,
            Prediksi=prediksi,
        )



        return JsonResponse({'status': 'success'})
    else:
        return JsonResponse({'status': 'error'})


