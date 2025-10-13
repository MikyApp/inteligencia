import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, export_text

# Datos
data_image = {
    'Edad': ['joven', 'joven', 'joven', 'joven', 'joven', 'adulto', 'adulto', 'adulto', 'adulto', 'adulto', 'adulto', 'adulto', 'adulto', 'joven', 'joven', 'adulto', 'adulto', 'joven', 'joven', 'adulto', 'joven'],
    'Profesional': ['si', 'si', 'no', 'si', 'no', 'si', 'no', 'si', 'no', 'si', 'no', 'si', 'no', 'si', 'si', 'no', 'no', 'no', 'no', 'si', 'si'],
    'Ingresos': ['bajos', 'altos', 'altos', 'bajos', 'medios', 'altos', 'altos', 'altos', 'medios', 'bajos', 'medios', 'medios', 'altos', 'altos', 'medios', 'medios', 'bajos', 'medios', 'bajos', 'medios', 'medios'],
    'Sexo': ['hombre', 'hombre', 'mujer', 'mujer', 'mujer', 'hombre', 'mujer', 'mujer', 'mujer', 'mujer', 'mujer', 'hombre', 'hombre', 'mujer', 'hombre', 'hombre', 'hombre', 'hombre', 'mujer', 'mujer', 'mujer'],
    'Interesado': ['si', 'si', 'no', 'si', 'no', 'no', 'no', 'no', 'no', 'no', 'no', 'no', 'si', 'si', 'si', 'no', 'no', 'no', 'no', 'no', 'si']
}

# Crear DataFrame
df_image = pd.DataFrame(data_image)

# Codificar variables categóricas
le_image = LabelEncoder()
for col in ['Edad', 'Profesional', 'Ingresos', 'Sexo', 'Interesado']:
    df_image[col] = le_image.fit_transform(df_image[col])

# Separar características y variable objetivo
X_image = df_image.drop(columns=['Interesado'])
y_image = df_image['Interesado']

# Implementar ID3 con criterio 'entropy'
id3_model_image = DecisionTreeClassifier(criterion='entropy')
id3_model_image.fit(X_image, y_image)

# Mostrar reglas del modelo ID3
rules_id3_image = export_text(id3_model_image, feature_names=list(X_image.columns))
print("Reglas ID3:\n", rules_id3_image)

# Implementar J48 aproximado usando poda (max_depth)
j48_model_image = DecisionTreeClassifier(criterion='entropy', max_depth=4)
j48_model_image.fit(X_image, y_image)

# Mostrar reglas del modelo J48 aproximado
rules_j48_image = export_text(j48_model_image, feature_names=list(X_image.columns))
print("Reglas J48 aproximado:\n", rules_j48_image)
