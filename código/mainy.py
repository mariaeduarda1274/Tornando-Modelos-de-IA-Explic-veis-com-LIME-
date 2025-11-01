## Importante, rodar cada código uma parte, não tudo de uma vez.


# Comando para instalar a biblioteca LIME
!pip install lime

# O Colab já deve ter os outros, mas você pode rodar para garantir:
# !pip install pandas numpy scikit-learn

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
# 6. VISUALIZAÇÃO PROFISSIONAL FINAL (VERSÃO DEFINITIVA E LIMPA)
# ==============================================================================
import matplotlib.pyplot as plt
import re

# 1. Mapeamento Semântico
category_map = {
    'A14': 'Sem Conta Corrente (Risco)',
    'A152': 'Conta com Saldo Positivo',
    'A31': 'Histórico: Pagamentos em dia',
    'A34': 'Histórico: Atraso em Pagamentos (Risco)',


    'A41': 'Propósito: Carro Novo',
    'A46': 'Propósito: Móveis / Equipamento',
}
attribute_map = {
    '12': 'Idade (anos)',
    '4': 'Monto do Crédito (DM)', # Correção para o índice 4
}

# 2. Obter a Explicação e Traduzir as Features
explanation_list = explanation.as_list()
translated_features = []
count = 0 # <<--- CORREÇÃO: Inicializa o contador

for feature_expression, weight in explanation_list:
    # 1. Parar o loop depois de 5 fatores CLAROS serem adicionados
    if count >= 5:
        break

    clean_expression = feature_expression
    traduzido_encontrado = False

    # 0. TRADUÇÃO SEMÂNTICA (Substitui a faixa numérica por uma frase simples)
    if '1380 < Monto do Crédito (DM) <=' in feature_expression:
        clean_expression = 'Valor de Crédito Moderado' # Opção: Manter o nome traduzido se não quiser a frase
        traduzido_encontrado = True

    # 1. Lógica de Tradução para Categóricas (códigos)
    if not traduzido_encontrado:
        for code, traducao in category_map.items():
            if code in feature_expression:
                clean_expression = traducao
                traduzido_encontrado = True
                break

    # 2. LÓGICA REFORÇADA PARA TRADUZIR VARIÁVEIS NUMÉRICAS COM NOTAÇÃO LIME
    if not traduzido_encontrado:
        for index_str, attr_name in attribute_map.items():
            if f' < {index_str} <=' in feature_expression or f' > {index_str} ' in feature_expression:
                clean_expression = feature_expression.replace(f' {index_str} ', f' {attr_name} ')
                traduzido_encontrado = True
                break

    # 3. Lógica de tradução de outros índices numéricos (A lógica antiga, mas mantida)
    if not traduzido_encontrado:
        match_index = re.match(r'(\d+)', feature_expression)
        original_index = match_index.group(1) if match_index else None

        if original_index in attribute_map:
             attribute_name = attribute_map[original_index]
             clean_expression = feature_expression.replace(original_index, attribute_name)
        else:
            clean_expression = feature_expression

    # 4. LIMPEZA FINAL GERAL: Remove todos os '.00'
    clean_expression = re.sub(r'(\d)\.00', r'\1', clean_expression)
    clean_expression = re.sub(r'(\d)\.0', r'\1', clean_expression)

    # NOVO FILTRO DE EXCLUSÃO
    # Exclui o fator de Monto do Crédito para buscar o 6º fator (que é mais claro)
    if 'Monto do Crédito' in clean_expression or 'Valor de Crédito Moderado' in clean_expression:
        continue # Pula este fator

    # Adicionar o fator traduzido
    translated_features.append((clean_expression, weight))
    count += 1


# 3. Desenhar o Gráfico Matplotlib
features = [item[0] for item in translated_features]
weights = [item[1] for item in translated_features]
colors = ['#dc3545' if w < 0 else '#28a745' for w in weights]

plt.style.use('fivethirtyeight')
fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(features, weights, color=colors, alpha=0.9)
ax.axvline(x=0, color='gray', linestyle='--', linewidth=1)

# TÍTULO FINAL DINÂMICO
# A. PREDIÇÃO (JÁ MUDADO PARA APROVADO/NEGADO)
prediction_label = 'Negado' if prediction == 0 else 'Aprovado'

# B. UNIFORMIZAÇÃO DOS TÍTULOS PRINCIPAIS
if 'Negado' in prediction_label:
    titulo_principal = 'TRANSPARÊNCIA LIME: FATORES PARA A NEGAÇÃO' # Mantido
else:
    # MUDA AQUI! Para usar a mesma estrutura "FATORES PARA..."
    titulo_principal = 'TRANSPARÊNCIA LIME: FATORES PARA A APROVAÇÃO'

# C. NOME FICTÍCIO (MANTIDO)
nome_cliente_ficticio = "Cliente Teste João"

# D. EXIBIÇÃO
ax.set_title(
    f'{titulo_principal}: {nome_cliente_ficticio} (ID {X_test.index[instance_index]})',
    fontsize=14,
    color='#343a40',
    fontweight='bold'
)
ax.set_xlabel('Impacto na Probabilidade de Ser "Bom Risco" (Peso)', fontsize=11, color='#495057')
ax.set_ylabel('Fatores Determinantes', fontsize=11, color='#495057')

plt.tick_params(axis='y', labelsize=10)
plt.gca().invert_yaxis()
plt.tight_layout()

# 4. SALVAR E FECHAR A IMAGEM (REMOVE WARNINGS/OUTPUTS)
nome_arquivo_definitivo = f'LIME_FINAL_{prediction_label.replace(" ", "_").upper()}_ID_{X_test.index[instance_index]}.png'
fig.savefig(nome_arquivo_definitivo, bbox_inches='tight', dpi=300)
plt.close(fig)

print(f"\n>>> IMAGEM FINAL SALVA E PRONTA PARA ENTREGA: '{nome_arquivo_definitivo}' <<<")

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
