#  📌Decifrando a Caixa Preta:Tornando Modelos de IA Explicáveis com LIME 
- Aplicar técnicas de Explainable AI (XAI)
## 1.Contextualização do Problema e Definição dos Objetivos

### - Contextualização do Desafio 

Este projeto se concentra em um desafio crucial para as empresas que usam modelos de Machine Learning na avaliação de crédito. Embora os modelos de Machine Learning (ML) atinjam excelentes índices de precisão, eles frequentemente operam como uma "caixa-preta".
Diante disso, clientes, gerentes e órgãos regulatórios passaram a questionar **"por que"** o modelo toma determinadas decisões, com foco especial nos casos de negação de crédito.Exige que a empresa seja capaz de explicar cada decisão individual de forma transparente, clara e tecnicamente fundamentada.

O problema central, é a necessidade de ir além da precisão: o objetivo é transformar um modelo que apenas acerta o resultado em um modelo que também seja explicável e transparente em cada decisão.

### - Definição dos Objetivos

A missão deste projeto é aplicar técnicas de **Explainable AI (XAI)**, utilizando a biblioteca **LIME** (Local Interpretable Model-agnostic Explanations), para solucionar o problema da opacidade.

Os objetivos específicos são:
1.  **Desenvolver** um modelo de classificação sobre o dataset **Statlog (German Credit Data)**.
2.  **Gerar explicações locais** (LIME) que mostrem quais características do cliente (como histórico de inadimplência, tipo de conta ou idade) mais impactaram na decisão do modelo para clientes analisados.
3.  **Demonstrar a transparência** das decisões do modelo, tanto em casos de **Negação** (Mau Risco) quanto em casos de **Aprovação** (Bom Risco), conforme exigido para fins de compliance e comunicação com o cliente.


### 2. Explicação do Modelo Preditivo Escolhido

Para o desafio de classificar clientes entre "Bom Risco" e "Mau Risco" (classificação binária), foi escolhido o algoritmo de **Random Forest (Floresta Aleatória)**.

#### Por que escolhi o Random Forest?
1.  **Alta Acurácia:** O Random Forest é um modelo de aprendizado de conjunto conhecido por entregar alta acurácia e estabilidade, sendo muito eficaz em problemas de classificação do mundo real, como o credit scoring.
2.  **Robustez:** Ele lida bem com a complexidade e a não-linearidade dos dados de crédito, além de ser menos suscetível a *overfitting* (sobreajuste) em comparação com uma única Árvore de Decisão.
3.  **O Problema da Caixa-Preta:** Apesar de sua excelente performance, o Random Forest opera como um modelo de **"caixa-preta"**. Sua decisão final é o resultado da média de centenas de árvores individuais, o que torna impossível para um ser humano rastrear ou explicar o "porquê" de uma decisão individual.

#### Papel do Modelo no Projeto
O modelo Random Forest serve como a **base preditiva** do projeto. Sua alta precisão valida a utilidade do sistema, enquanto sua opacidade (a "caixa-preta") justifica a necessidade e a aplicação da ferramenta LIME. A explicação do LIME é, portanto, o método de tornar as previsões deste modelo complexo transparentes para o cliente e para fins de auditoria.
