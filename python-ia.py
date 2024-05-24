import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

# Carregar o conjunto de dados MNIST
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

# Normalizar os valores dos pixels para o intervalo [0, 1]
train_images, test_images = train_images / 255.0, test_images / 255.0

# Construir o modelo da rede neural
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)), # Converter cada imagem 28x28 em um vetor de 784 elementos
    layers.Dense(128, activation='relu'), # Camada densa com 128 neurônios e função de ativação ReLU
    layers.Dense(10, activation='softmax') # Camada de saída com 10 neurônios e função de ativação softmax
])

# Compilar o modelo
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Treinar o modelo
history = model.fit(train_images, train_labels, epochs=5, 
                    validation_data=(test_images, test_labels))

# Avaliar o modelo
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(f'\nAcurácia no conjunto de testes: {test_acc}')

# Plotar a acurácia de treinamento e validação ao longo do tempo
plt.plot(history.history['accuracy'], label='Acurácia de Treinamento')
plt.plot(history.history['val_accuracy'], label='Acurácia de Validação')
plt.xlabel('Época')
plt.ylabel('Acurácia')
plt.legend(loc='lower right')
plt.show()