# Reconhecimento de Dígitos Manuscritos com Redes Neurais  

Este projeto implementa uma rede neural **MLP (Multilayer Perceptron)** para reconhecer dígitos manuscritos do dataset MNIST, utilizando aprendizado supervisionado com correção de erro (*backpropagation*).  

## 📌 Técnicas e Resultados  
- **Acurácia no teste**: acima de 90% (dados nunca vistos).  
- **Técnicas anti-overfitting**: Validação cruzada (`validation_split=0.2`) e *EarlyStopping*.  
- **Tempo de treinamento**: ~10 épocas (com parada antecipada).
- **Criação do modelo**: uso de funções de ativação: ReLU e Sotfmax.

## 🛠️ Tecnologias  
- **Linguagem**: Python 3.8+  
- **Bibliotecas**: TensorFlow, Keras, NumPy, Matplotlib  
- **Dataset**: MNIST (60k imagens de treino + 10k de teste).  
