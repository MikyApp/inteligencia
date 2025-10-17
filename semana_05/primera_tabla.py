import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.tree import DecisionTreeClassifier, export_text

# Datos
data = {
    'estado-civil': ['soltero', 'soltero', 'soltero', 'casada', 'casada', 'casada', 'soltera', 'casado', 'soltero', 'casado', 'casado', 'soltero', 'soltero', 'casado', 'casado', 'viudo', 'divorciado'],
    'ocupacion': ['estudiante', 'estudiante', 'estudiante', 'estudiante', 'estudiante', 'estudiante', 'trabaja', 'trabaja', 'trabaja', 'trabaja', 'trabaja', 'estudiante', 'estudiante', 'estudiante', 'trabaja', 'trabaja', 'trabaja'],
    'monto': [1500, 1700, 1800, 3500, 2500, 1800, 4000, 5000, 3700, 2500, 6000, 5000, 300, 6500, 800, 1200, 2450],
    'ocup-garante': ['empl-publico', 'secretaria', 'profesor', 'doctor', 'empl-publico', 'gerente', 'contador', 'fotografo', 'zapatero', 'comerciante', 'panadero', 'rela-publico', 'ejecutivo', 'empl-publico', 'empl-publico', 'profesor', 'profesor'],
    'sueldo-gar': [800, 250, 500, 560, 750, 800, 960, 670, 200, 270, 579, 300, 700, 600, 500, 400, 600],
    'otorga-credito': ['no', 'no', 'no', 'si', 'no', 'si', 'si', 'si', 'si', 'no', 'no', 'si', 'no', 'no', 'si', 'si', 'no']
}

# Crear DataFrame
df = pd.DataFrame(data)

# Codificar variables categóricas
le = LabelEncoder()
for col in ['estado-civil', 'ocupacion', 'ocup-garante', 'otorga-credito']:
    df[col] = le.fit_transform(df[col])

# Separar características y variable objetivo
target = 'otorga-credito'
X = df.drop(columns=[target])
y = df[target]

# Modelo ID3 con criterio Entropía
id3_model = DecisionTreeClassifier(criterion='entropy')
id3_model.fit(X, y)

# Extraer reglas del árbol
rules = export_text(id3_model, feature_names=list(X.columns))
print(rules)



# Entrenar modelo J48 aproximado con poda mediante max_depth para limitar complejidad
j48_model = DecisionTreeClassifier(criterion='entropy', max_depth=4)
j48_model.fit(X, y)

# Extraer reglas del árbol J48 aproximado
rules_j48 = export_text(j48_model, feature_names=list(X.columns))
print(rules_j48)
