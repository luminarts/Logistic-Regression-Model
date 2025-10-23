# Modelo de Classificação com Regressão Logística
## Objetivo

Esse código objetiva demonstrar uma implementação manual do modelo de machine learning de aprendizado supervisionado de classificação com regressão logística. O que está sendo apresentado aqui é somente um estudo embasado em técnicas aprendidas em cursos e livros, e não possui nenhum fim comercial.

---

## Descrição do Código  

O modelo foi treinado utilizando o **dataset MNIST**, composto por imagens de dígitos manuscritos (0–9) em grayscale (1 canal de cor).  

### Pré-processamento e Normalização  
- Inicialmente, os pixels foram normalizados para o intervalo [0, 1], dividindo seus valores por 255.  
- Em seguida, foi aplicada uma normalização por média 0 e desvio padrão 1, de forma a centralizar e padronizar os dados.  
  Essa etapa garante uma convergência mais estável durante o treinamento e evita que variáveis com escalas diferentes dominem o aprendizado.  

### Função de Erro (Loss Function)  
A função de custo utilizada foi a **Categorical Cross-Entropy Loss (CCE)**, amplamente utilizada em classificações multiclasse.  
Ela mede a diferença entre as probabilidades previstas pelo modelo e as classes verdadeiras, penalizando predições incorretas com maior intensidade.  

### Regularização L2  
Para evitar *overfitting*, foi implementada **regularização L2 (Ridge)**, que adiciona um termo penalizador proporcional ao quadrado dos pesos do modelo.  
Isso ajuda a manter os coeficientes mais próximos de zero, promovendo uma generalização melhor em dados não vistos.  

### Função Softmax  
A camada de saída utiliza a **função Softmax**, responsável por converter os valores lineares da regressão logística em **probabilidades normalizadas** entre 0 e 1, garantindo que a soma das saídas para todas as classes seja igual a 1.  

### Validação Cruzada (K-Fold Cross Validation)  
Para avaliar o desempenho do modelo de forma robusta, foi utilizada a técnica de **K-Fold Cross Validation**, que divide o conjunto de treinamento em *k* partições (folds).  
O modelo é treinado e validado múltiplas vezes, alternando as partições, garantindo uma estimativa mais confiável da performance média e reduzindo o viés na avaliação.  

---

## Resultados
(Botar gráficos e descrever melhor learning rates e epochs ideiais, além do melhor lambda de regularização)
(Falar dos metodos do evaluate: acurácia, f1-score e confusion matrix)

## Pontos de Melhoria Futura
Algumas melhorias planejadas para versões futuras incluem:  

- **Implementação de novos métodos de otimização**, como:  
  - **Stochastic Gradient Descent (SGD)**  
  - **Mini-Batch Gradient Descent**  
  - **AdamW** e/ou outros otimizadores modernos para acelerar a convergência  

- **Aceleração via CUDA:**  
  Adaptação do código para execução em **GPU**, utilizando *PyTorch CUDA tensors* para reduzir significativamente o tempo de treinamento.  

- **Interface Gráfica Interativa:**  
  Desenvolvimento de uma **interface visual** que permita ao usuário **desenhar um dígito à mão livre** e obter a **previsão do modelo em tempo real**, tornando o projeto mais intuitivo e acessível.  

---

## ⚙️ Como Executar o Código  

### 1. Clonar o Repositório  
```bash
git clone https://github.com/usuario/nome-do-repositorio.git
cd nome-do-repositorio
```

### 2. Criar e ativar o seu ambiente virtual
```bash
python -m venv myvenv
source myvenv/bin/activate # para Linux/Mac
myvenv/Scripts/Activate # para Windows
```

### 3. Instalar dependências

```bash
pip install -r libs.txt
```

#### Execute o programa!