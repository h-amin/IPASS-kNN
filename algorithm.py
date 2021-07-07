import csv
import math


# ----------------------------------- TO-DO ------------------------------------------------
# Zorg ervoor dat de classificaties worden meegenomen in csv_extract, om ervoor te
# zorgen dat je nn_lst aangeeft van welk classificatie het toe behoort.
# Vervolgens dat data meegeven aan je result en laat merken dat de classificatie correct is.
# ------------------------------------------------------------------------------------------


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
    train_data = [list(map(float, data_str[k])) for k in range(len(data_str))]
    return train_data


# Formule om de Euclidean afstand te berekenen m.b.v. vectors
def euclidean_distance_formula(x1, x2):
    afstand = 0.0
    for i in range(len(x1) - 1):
        afstand += (x1[i] - x2[i]) ** 2
    sqrt_afstand = math.sqrt(afstand)
    print(sqrt_afstand)
    return sqrt_afstand


# Functie dat de dichtstbijzijnde waarden vindt.
def find_NN(train_data, test, nn):
    afstand_lst = []
    for record in train_data:
        ecld_product = euclidean_distance_formula(test, record)
        afstand_lst.append((record, ecld_product))
    afstand_lst.sort(key=lambda result: result[1])
    nn_lst = []
    for i in range(nn):
        nn_lst.append(afstand_lst[i][0])

    return nn_lst


# Classificatie voorspelling m.b.v. find_NN. (NN = nearest_neighbor)
def predict_class(train_data, test, nn):
    nearest_neighbors = find_NN(train_data, test, nn)
    x = [i[-1] for i in nearest_neighbors]

    voorspelling = max(set(x), key=x.count)

    voorspelling = math.ceil(voorspelling)
    # print(voorspelling)
    return voorspelling
