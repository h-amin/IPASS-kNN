import csv
import math
from collections import Counter


# Uitpakken van het csv bestand
def csv_extract(file_names):
    data_lst = []
    with open(file_names, 'r') as f:
        csv_r = csv.reader(f)
        for record in csv_r:
            if not record:
                continue
            data_lst.append(record)

    global class_lst
    class_lst = [data_lst[i][-1] for i in range(len(data_lst))]  # Lijst met alle classificaties.

    # Verwijderen van iris classificatie
    data_str = [data_lst[h][:4] for h in range(len(data_lst))]  # Lijst met alle numerieke waarden.

    # Str waarde naar float veranderen
    train_data = [list(map(float, data_str[k])) for k in range(len(data_str))]

    return train_data


# Formule om de Euclidean afstand te berekenen m.b.v. vectors
def euclidean_distance_formula(x1, x2):
    afstand = 0.0
    for i in range(len(x1) - 1):
        afstand += (x1[i] - x2[i]) ** 2
    sqrt_afstand = math.sqrt(afstand)
    return sqrt_afstand


# Functie dat de dichtstbijzijnde waarden vindt.
def find_NN(train_data, test, nn):
    afstand_lst = []
    for record in train_data:
        ecld_product = euclidean_distance_formula(test, record)
        afstand_lst.append((record, ecld_product))

    afstand_lst.sort(key=lambda x: x[1])  # Dit methode sorteert de afstand_lst op kleinste ecld_product waarde.

    nn_lst = [afstand_lst[i][0] for i in range(nn)]  # record list met n aantal records afhankelijk van de nn waarde.

    return nn_lst


# Classificatie voorspelling m.b.v. find_NN. (NN = nearest_neighbor)
def predict_class(train_data, test, nn):
    nearest_neighbors = find_NN(train_data, test, nn)
    classification_nn = []

    for i in range(len(nearest_neighbors)):
        element = nearest_neighbors[i]
        index = train_data.index(element) + 1
        classification = class_lst[index - 1]
        classification_nn.append(classification)

    counts = Counter(classification_nn)  # Een dictionary dat de classes telt en in key/values opslaat voor hergebruik.

    max_value = max(counts.values())

    product = [k for k, v in counts.items() if v == max_value]  # vindt het maximum value en alle keys die
    # daarop corresponderen.

    for j in range(len(product)):
        voorspelling = product[j]
        return voorspelling
