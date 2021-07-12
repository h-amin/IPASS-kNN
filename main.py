import algorithm

# Voorspelling maken met het algoritme
file_name = 'iris.csv'
train_data = algorithm.csv_extract(file_name)
# k = n parameter, aangeven wat de span is van het algoritme.
n_range = 5  # nn
# Hier vul je steeds het nieuwe record in waarvan je het classificatie wilt weten.
row = [4.1, 3.5, 1.2, 2.5]
# Voorspellingsmodel
label = algorithm.predict_class(train_data, row, n_range)
print('Voor de record %s,  wordt er een voorspelling gedaan met waarde: %s' % (row, label))
