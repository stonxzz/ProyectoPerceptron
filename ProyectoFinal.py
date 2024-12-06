import numpy as np

# Tabla de verdad para una puerta lógica (OR en este caso)
def generar_datos():
    X = np.array([
        [1, 1, 1],   # X0, X1, X2
        [1, 1, -1],
        [1, -1, 1],
        [1, -1, -1]
    ])
    y_d = np.array([-1, 1, 1, -1])  # Salida esperada (deseada)
    return X, y_d

# Inicialización de pesos y parámetros
def inicializar_pesos():
    return np.random.rand(3)  # Pesos iniciales aleatorios

# Función de activación: Signo
def funcion_activacion(valor):
    return 1 if valor >= 0 else -1

# Función para imprimir resultados de cada iteración
def imprimir_resultados(epoch, X, y_d, y_pred, error, w):
    print(f"\nIteración {epoch + 1}:")
    for i in range(len(X)):
        print(f"  Entrada: {X[i]}, Salida esperada: {y_d[i]}, Predicha: {y_pred[i]}, Error: {error[i]}")
        if error[i] != 0:
            print(f"    Pesos actualizados: {w}")

# Entrenamiento del perceptrón
def entrenar_perceptron(X, y_d, w, alpha, iteraciones):
    for epoch in range(iteraciones):
        y_pred = []
        error_total = 0
        error = []
        for i in range(len(X)):
            # Cálculo de la salida del perceptrón
            y = np.dot(w, X[i])  # Producto punto (sumatoria ponderada)
            y_pred.append(funcion_activacion(y))  # Salida activada

            # Cálculo del error
            error.append(y_d[i] - y_pred[i])
            error_total += abs(error[i])

            # Actualización de pesos si hay error
            if error[i] != 0:
                w += alpha * error[i] * X[i]

        imprimir_resultados(epoch, X, y_d, y_pred, error, w)

        # Detener si el error total es cero (solución encontrada)
        if error_total == 0:
            print("\nEl perceptrón ha aprendido correctamente los pesos.")
            break
    else:
        print("\nNo se alcanzó un error total de cero en las iteraciones permitidas.")

    return w

# Función principal
def main():
    X, y_d = generar_datos()
    w = inicializar_pesos()
    alpha = 0.4  # Tasa de aprendizaje
    iteraciones = 10  # Número máximo de iteraciones

    print("Pesos iniciales:", w)
    w_final = entrenar_perceptron(X, y_d, w, alpha, iteraciones)
    print("\nPesos finales:", w_final)

if __name__ == "__main__":
    main()
