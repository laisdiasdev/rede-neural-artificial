# Importação das bibliotecas necessárias
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import numpy as np

# Carregando o conjunto de dados MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalizando os dados para ficar entre 0 e 1
x_train = x_train / 255.0
x_test = x_test / 255.0

# Exibindo o formato dos dados para conferência
print("Formato do x_train:", x_train.shape)
print("Formato do y_train:", y_train.shape)

# Definindo a estrutura da rede neural
modelo = Sequential([
    Flatten(input_shape=(28, 28)),  # Camada de entrada: 784 neurônios (28x28 pixels)
    Dense(128, activation='relu'),  # Camada oculta: 128 neurônios com ReLU
    Dense(10, activation='softmax') # Camada de saída: 10 neurônios para as classes 0 a 9
])

# Compilando o modelo
modelo.compile(optimizer='adam',                 # Otimizador baseado em Gradiente Descendente
              loss='sparse_categorical_crossentropy',  # Função de perda para classificação
              metrics=['accuracy'])              # Métrica de desempenho


# Definindo parada antecipada para evitar overfitting
parada_antecipada = EarlyStopping(monitor='val_loss', patience=3)

# Treinando a rede neural
historico = modelo.fit(x_train, y_train,
                    epochs=20,
                    validation_split=0.2,
                    callbacks=[parada_antecipada])  # Aplicação da validação por divisão de 20% dos dados de treino


# Avaliando o modelo com dados de teste
perda_test, acuracia_test = modelo.evaluate(x_test, y_test)
print('Acurácia no conjunto de teste:', acuracia_test)

# Plotando a evolução da acurácia durante o treinamento
plt.figure(figsize=(8,5))
plt.plot(historico.history['accuracy'], label='Acurácia de Treinamento')
plt.plot(historico.history['val_accuracy'], label='Acurácia de Validação')
plt.title('Evolução da Acurácia')
plt.xlabel('Épocas')
plt.ylabel('Acurácia')
plt.legend()
plt.grid()
plt.show()


previsao = modelo.predict(x_test)

# Mostrar uma imagem com a previsão
indice = int(input("Digite um índice entre 0 e 9 para visualizar a imagem de teste: "))
previsao = modelo.predict(x_test[indice].reshape(1, 28, 28))
plt.imshow(x_test[indice], cmap='gray')
plt.title(f"Valor real: {y_test[indice]} | Predito: {np.argmax(previsao)}")
plt.axis('off')
plt.show()
