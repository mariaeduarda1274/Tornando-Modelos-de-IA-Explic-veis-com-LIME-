## Importante, rodar cada código uma parte, não tudo de uma vez.
Instruções para o código não da erro:
️- Reinicie o ambiente antes (menu → Ambiente de execução → Reiniciar ambiente de execução)
Isso limpa tudo e evita variáveis “velhas”.

- Depois, rode cada célula na ordem, assim:
1. Instalar bibliotecas (pip install lime)
2. Carregar e preparar o dataset (german.data)
3. Fazer o encoding (get_dummies)
4. Treinar o modelo (RandomForest)
5. Bloco 5: LIME explicação ID 740
6. Bloco 6: Função gerar gráfico + gerar ambos gráficos (521 e 740)

- Verifique se o arquivo german.data está no Colab.

!pip install lime

import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from lime.lime_tabular import LimeTabularExplainer

print("Bibliotecas importadas com sucesso!")

# CÉLULA DE CÓDIGO DO COLAB
import pandas as pd
import numpy as np

# 1. Carregar o Dataset 'german.data'
# Ele usa espaço (' ') como separador e não tem cabeçalho (header=None), pois não tem nomes de coluna
file_name = 'german.data'
df = pd.read_csv(file_name, sep=' ', header=None)

# 2. Renomear a Coluna Alvo (Target)
# O dataset tem 20 atributos de entrada (colunas 0 a 19) e a última coluna (coluna 20)
# é a variável alvo: 1 = Bom Risco, 2 = Mau Risco.
df.rename(columns={20: 'Credit_Risk'}, inplace=True)

# 3. Converter a Coluna Alvo para 0 e 1 (Padrão de ML)
# Vamos converter o 2 (Mau Risco) para 0 e o 1 (Bom Risco) para 1.
# (0 = Mau Risco, 1 = Bom Risco)
df['Credit_Risk'] = df['Credit_Risk'].apply(lambda x: 1 if x == 1 else 0)

# 4. Inspecionar o resultado
print("-----------------------------------------")
print("Dimensão do Dataset:", df.shape)
print("Contagem da Variável Alvo (Target):")
print(df['Credit_Risk'].value_counts()) # Ver quantos 'bons' (1) e 'maus' (0) riscos existem
print("-----------------------------------------")
print(df.head())


# 1. Identificar as colunas de atributos (todas, exceto a última)
# O dataset tem 21 colunas (0 a 20). A coluna 20 é 'Credit_Risk', as de entrada são 0 a 19.
X = df.drop('Credit_Risk', axis=1) # Atributos (features)
y = df['Credit_Risk']             # Variável Alvo (target)

# 2. Aplicar One-Hot Encoding em TODAS as colunas de atributos (X)
# Esta função do Pandas identifica automaticamente colunas não-numéricas e as converte.
# Exemplo: A coluna 0 (Account Status) terá novas colunas como '0_A11', '0_A12', etc.
X_encoded = pd.get_dummies(X, drop_first=True)

# 3. Inspecionar o resultado do Encoding
print("-----------------------------------------")
print("Dimensão dos Atributos ANTES do Encoding:", X.shape)
print("Dimensão dos Atributos DEPOIS do Encoding:", X_encoded.shape)
print("-----------------------------------------")
print("Primeiras linhas dos atributos codificados (X_encoded.head()):")
print(X_encoded.head())

# Agora X_encoded contém apenas números!


# CÉLULA DE CÓDIGO 3
# 1. Identificar as colunas de atributos
X = df.drop('Credit_Risk', axis=1) # Atributos (features)
y = df['Credit_Risk']             # Variável Alvo (target)

# 2. Aplicar One-Hot Encoding
X_encoded = pd.get_dummies(X, drop_first=True)

# 3. Converter todos os nomes de colunas (features) para STRING
X_encoded.columns = X_encoded.columns.astype(str)

# 4. Inspecionar o resultado (Este print pode ser mantido ou deletado, se quiser)
print("-----------------------------------------")
print("Dimensão dos Atributos DEPOIS do Encoding:", X_encoded.shape)
# ... o restante dos prints

# CÉLULA DE CÓDIGO 4: TREINAMENTO DO RANDOM FOREST

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# 1. Separar os dados em conjuntos de Treino e Teste
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42
)

# 2. Criar e Treinar o Modelo (Random Forest Classifier)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 3. Avaliar o Desempenho do Modelo no conjunto de Teste
y_pred = model.predict(X_test)

print("-----------------------------------------")
print("RELATÓRIO DE CLASSIFICAÇÃO (Performance do Modelo):")
print(classification_report(y_test, y_pred))
print("-----------------------------------------")


!pip install lime


# ==============================================================================
# 5. APLICAÇÃO DO LIME (USANDO O ÍNDICE DEFINIDO NO INÍCIO OU NO BLOCO 15)
# ==============================================================================
# FIX: FORÇA A SELEÇÃO DO CLIENTE DE NEGAÇÃO (ID 740) ANTES DO CÁLCULO LIME
ID_PARA_EXPLICAR = 740

# Estas linhas SOBRESCREVEM qualquer valor anterior de instance_index
instance_index = X_test.index.get_loc(ID_PARA_EXPLICAR)
instance_data = X_test.iloc[instance_index].values
instance_label = y_test.iloc[instance_index]
# Importar a biblioteca LIME
from lime.lime_tabular import LimeTabularExplainer
import numpy as np # Garante que numpy está disponível
import pandas as pd # Garante que pandas está disponível

# 1. Definir o Explainer do LIME
explainer = LimeTabularExplainer(
    training_data=np.array(X_train),
    feature_names=X_encoded.columns.tolist(),
    class_names=['Mau Risco', 'Bom Risco'],
    mode='classification',
    random_state=42
)

# 3. Gerar a Explicação LIME para a Instância
explanation = explainer.explain_instance(
    data_row=instance_data,
    predict_fn=model.predict_proba,
    num_features=8
)
# 4. Exibir o Resultado (Texto) e definir prediction_label (para o Bloco 6)
prediction = model.predict(instance_data.reshape(1, -1))[0]
prediction_label = 'Mau Risco' if prediction == 0 else 'Bom Risco' # Define para o Bloco 6!

print(f"-----------------------------------------")
print(f"CLIENTE A SER EXPLICADO (ID {X_test.index[instance_index]}):")
print(f"Decisão REAL (True Label): {'Bom Risco' if instance_label == 1 else 'Mau Risco'}")
print(f"PREVISÃO do Modelo: {prediction_label}")
print(f"-----------------------------------------")
print(f"Explicação LIME em texto (5 Fatores mais importantes):")
print(explanation.as_list())


# ==============================================================================  
# 6. VISUALIZAÇÃO PROFISSIONAL FINAL - GERAR GRÁFICOS DE APROVAÇÃO E NEGAÇÃO
# ==============================================================================

import matplotlib.pyplot as plt
import re

def gerar_grafico_lime(ID_PARA_EXPLICAR):
    """Função que gera e salva o gráfico LIME para um cliente específico."""
    
    # Localizar a instância
    instance_index = X_test.index.get_loc(ID_PARA_EXPLICAR)
    instance_data = X_test.iloc[instance_index].values
    instance_label = y_test.iloc[instance_index]
    
    # Gerar explicação
    explanation = explainer.explain_instance(
        data_row=instance_data,
        predict_fn=model.predict_proba,
        num_features=8
    )
    prediction = model.predict(instance_data.reshape(1, -1))[0]
    prediction_label = 'Negado' if prediction == 0 else 'Aprovado'

    # Mapeamentos semânticos
    category_map = {
        'A14': 'Sem Conta Corrente (Risco)',
        'A152': 'Conta com Saldo Positivo',
        'A31': 'Histórico: Pagamentos em dia',
        'A34': 'Histórico: Atraso em Pagamentos (Risco)',
        'A41': 'Propósito: Carro Novo',
        'A46': 'Propósito: Móveis / Equipamento',
    }
    attribute_map = {'12': 'Idade (anos)', '4': 'Monto do Crédito (DM)'}

    explanation_list = explanation.as_list()
    translated_features = []
    count = 0

    for feature_expression, weight in explanation_list:
        if count >= 5:
            break
        clean_expression = feature_expression
        traduzido_encontrado = False

        if '1380 < Monto do Crédito (DM) <=' in feature_expression:
            clean_expression = 'Valor de Crédito Moderado'
            traduzido_encontrado = True

        if not traduzido_encontrado:
            for code, traducao in category_map.items():
                if code in feature_expression:
                    clean_expression = traducao
                    traduzido_encontrado = True
                    break

        if not traduzido_encontrado:
            for index_str, attr_name in attribute_map.items():
                if f' < {index_str} <=' in feature_expression or f' > {index_str} ' in feature_expression:
                    clean_expression = feature_expression.replace(f' {index_str} ', f' {attr_name} ')
                    traduzido_encontrado = True
                    break

        clean_expression = re.sub(r'(\d)\.00', r'\1', clean_expression)
        clean_expression = re.sub(r'(\d)\.0', r'\1', clean_expression)

        if 'Monto do Crédito' in clean_expression or 'Valor de Crédito Moderado' in clean_expression:
            continue

        translated_features.append((clean_expression, weight))
        count += 1

    features = [item[0] for item in translated_features]
    weights = [item[1] for item in translated_features]
    colors = ['#dc3545' if w < 0 else '#28a745' for w in weights]

    plt.style.use('fivethirtyeight')
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(features, weights, color=colors, alpha=0.9)
    ax.axvline(x=0, color='gray', linestyle='--', linewidth=1)

    titulo_principal = 'TRANSPARÊNCIA LIME: FATORES PARA A APROVAÇÃO' if prediction_label == 'Aprovado' else 'TRANSPARÊNCIA LIME: FATORES PARA A NEGAÇÃO'
    nome_cliente_ficticio = "Cliente Teste João"

    ax.set_title(f'{titulo_principal}: {nome_cliente_ficticio} (ID {ID_PARA_EXPLICAR})',
                 fontsize=14, color='#343a40', fontweight='bold')
    ax.set_xlabel('Impacto na Probabilidade de Ser "Bom Risco" (Peso)', fontsize=11, color='#495057')
    ax.set_ylabel('Fatores Determinantes', fontsize=11, color='#495057')
    plt.tick_params(axis='y', labelsize=10)
    plt.gca().invert_yaxis()
    plt.tight_layout()

    nome_arquivo_definitivo = f'LIME_FINAL_{prediction_label.upper()}_ID_{ID_PARA_EXPLICAR}.png'
    fig.savefig(nome_arquivo_definitivo, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"✅ IMAGEM SALVA: {nome_arquivo_definitivo}")

# ==================================================================
# GERAR AMBOS OS GRÁFICOS (APROVADO E NEGADO)
# ==================================================================
gerar_grafico_lime(521)  # Cliente aprovado
gerar_grafico_lime(740)  # Cliente negado

# ==============================================================================
# 15. ENCONTRAR UM CLIENTE NEGADO (MAU RISCO)
# ==============================================================================

# 1. Fazer as previsões no conjunto de teste
y_pred_test = model.predict(X_test)

# 2. Encontrar o índice do primeiro cliente que o modelo classificou como Mau Risco (0)
# Vamos garantir que ele também tenha sido classificado como Mau Risco na vida real, se possível,
# mas o mais importante é a PREVISÃO do modelo.
# np.where retorna uma tupla, pegamos o primeiro elemento que contém os índices.
negado_indices = np.where(y_pred_test == 0)[0]

if len(negado_indices) > 0:
    # O novo índice a ser analisado será o primeiro da lista
    novo_instance_index = negado_indices[0]

    print(f"ENCONTRADO: O modelo previu Mau Risco para o cliente de índice interno: {novo_instance_index}")
    print(f"O ID original desse cliente é: {X_test.index[novo_instance_index]}")

    # SALVAR ESTE NOVO ÍNDICE
    # Você precisará desta variável para rodar o Bloco 5 e 6 novamente.
    # Vamos redefinir 'instance_index' com o novo valor para os próximos blocos.
    instance_index = novo_instance_index

else:
    print("AVISO: Nenhum cliente Mau Risco (0) foi encontrado no conjunto de teste. O modelo pode estar superestimando o Bom Risco.")

# 3. Mostrar a Nova Previsão
instance_data = X_test.iloc[instance_index].values
prediction = model.predict(instance_data.reshape(1, -1))[0]
prediction_label = 'Mau Risco' if prediction == 0 else 'Bom Risco'

print(f"Próxima análise será para o ID {X_test.index[instance_index]} com previsão de {prediction_label}.")

# PASSO 1: FORÇAR A SELEÇÃO DO CLIENTE DE NEGAÇÃO (ID 740)

ID_DO_CLIENTE_NEGADO = 740
print(f"Selecionando cliente de negação ID: {ID_DO_CLIENTE_NEGADO}")

# Esta linha é o que realmente define a posição do cliente 740
instance_index = X_test.index.get_loc(ID_DO_CLIENTE_NEGADO)

# PASSO 2: RE-DEFINIR AS VARIÁVEIS USADAS PELO LIME
# Isso garante que a explicação seja gerada para o cliente 740 e não para o 521
instance_data = X_test.iloc[instance_index].values
instance_label = y_test.iloc[instance_index] 
