import algorithm

# Voorspelling maken met het algoritme
file_name = 'iris.csv'
data_set = algorithm.csv_extract(file_name)
# k = n parameter, aangeven wat de span is van het algoritme.
n_range = 5
# Hier vul je steeds het nieuwe record in waarvan je het classificatie wilt weten.
row = [7.1, 3.0, 5.9, 2.1]
# Voorspellingsmodel
label = algorithm.predict_class(data_set, row, n_range)
print('1 = setosa, 2 = versicolor, 3 = virginica')
print('Voor de record %s,  wordt er een voorspelling gedaan met waarde: %s' % (row, label))
