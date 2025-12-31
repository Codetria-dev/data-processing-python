import sys
import pandas as pd
import numpy as np
import matplotlib
import sklearn

print("Ambiente de desenvolvimento carregado com sucesso!")
print(f"Python: {sys.version}")
print(f"pandas: {pd.__version__}")
print(f"numpy: {np.__version__}")
print(f"matplotlib: {matplotlib.__version__}")
print(f"sklearn: {sklearn.__version__}")

#Teste básico 

df = pd.DataFrame({'col' : [1,2,3]})
print("DataFrame testado!")

