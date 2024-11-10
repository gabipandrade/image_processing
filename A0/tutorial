import imageio.v3 as imageio
import numpy as np

# Solicita ao usuário para inserir o nome do arquivo
filename = str(input())
filename = filename.strip()

# Constrói o caminho absoluto para o arquivo dentro do diretório /root

# Tenta carregar a imagem usando o caminho absoluto
try:
    img = imageio.imread(filename)
    # Solicita ao usuário as coordenadas do pixel
    n1 = int(input())
    n2 = int(input())
    # Acessa o valor do pixel nas coordenadas especificadas
    pixel = img[n1, n2]
    # Imprime o valor do pixel
    print(f"{pixel[0]} {pixel[1]} {pixel[2]}")
except FileNotFoundError:
    print(f"Arquivo não encontrado: {filename}")
except Exception as e:
    print(f"Ocorreu um erro: {e}")


