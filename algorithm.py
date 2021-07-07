import csv
import math


# Uitpakken van het csv bestand
def csv_extract(file_names):
    data_lst = []
    with open(file_names, 'r') as f:
        csv_r = csv.reader(f)
        for record in csv_r:
            if not record:
                continue
            data_lst.append(record)

    # Verwijderen van iris classificatie
    data_str = [data_lst[h][:4] for h in range(len(data_lst))]

    # Str waarde naar float veranderen
    data_set = [list(map(float, data_str[k])) for k in range(len(data_str))]
    return data_set


# Formule om de Euclidean afstand te berekenen m.b.v. vectors
def euclidean_distance_formula(x1, x2):
    afstand = 0.0
    for i in range(len(x1) - 1):
        afstand += (x1[i] - x2[i]) ** 2
    sqrt_afstand = math.sqrt(afstand)
    return sqrt_afstand


# Functie dat de dichtstbijzijnde waarden vindt.
def find_NN(data_set, test_row, nn):
    afstand_lst = []
    print(afstand_lst)
    for record in data_set:
        ecld_product = euclidean_distance_formula(test_row, record)
        afstand_lst.append((record, ecld_product))
    afstand_lst.sort(key=lambda result: result[1])
    print(afstand_lst)
    nn_lst = []
    for i in range(nn):
        nn_lst.append(afstand_lst[i][0])
    return nn_lst


# Classificatie voorspelling m.b.v. find_NN. (NN = nearest_neighbor)
def predict_class(data_set, test, nn):
    nearest_neighbors = find_NN(data_set, test, nn)
    x = [i[-1] for i in nearest_neighbors]
    voorspelling = max(set(x), key=x.count)
    voorspelling = math.ceil(voorspelling)
    return voorspelling


# Voorspelling maken met het algoritme
file_name = 'iris.csv'
data_set = csv_extract(file_name)
# k = n parameter, aangeven wat de span is van het algoritme.
n_range = 5
# Hier vul je steeds het nieuwe record in waarvan je het classificatie wilt weten.
row = [5.8,2.7,3.9,1.2]
# Voorspellingsmodel
label = predict_class(data_set, row, n_range)
print('1 = setosa, 2 = versicolor, 3 = virginica')
print('Voor de record %s,  wordt er een voorspelling gedaan met waarde: %s' % (row, label))
