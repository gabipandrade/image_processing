#Gabriela Passos de Andrade-12625142
import numpy as np
import imageio.v3 as imageio

def rmse(imgref, imgminha): #função de comparação entre imagem de refêrencia e imagem modificada
    N = imgref.shape #tamanho da imagem
    soma = 0.0 #inicializa a soma
    imgref = imgref.astype(np.float64)#converte para float o tipo da imagem
    imgminha = imgminha.astype(np.float64)#converte para float o tipo da imagem 
    for i in range(N[0]):
        for j in range(N[1]):#percorre a imagem
            erro_quadrado = (imgref[i, j] - imgminha[i, j]) ** 2#calcula o erro quadrado
            soma += erro_quadrado#soma o erro quadrado
    rmse = np.sqrt(soma / (N[0] * N[1]))#tira a raiz e calcula o rmse
    return rmse # retorna resultado

def single_image_histogram_equalization(img):#função de equalização de histograma
    hist, bins = np.histogram(img.flatten(), 256, [0, 256])#calcula o histograma da imagem
    cumulative_distribution_function = hist.cumsum()#calcula a função de distribuição cumulativa
    cumulative_distribution_function_m = np.ma.masked_equal(cumulative_distribution_function, 0)#mascara os valores iguais a 0
    cumulative_distribution_function_m = (cumulative_distribution_function_m - cumulative_distribution_function_m.min()) * 255 / (cumulative_distribution_function_m.max() - cumulative_distribution_function_m.min())#normaliza a função de distribuição cumulativa
    cumulative_distribution_function = np.ma.filled(cumulative_distribution_function_m, 0).astype('uint8')#preenche os valores da função de distribuição cumulativa e converte para uint8
    img_equalized = cumulative_distribution_function[img]#aplica a equalização de histograma
    return img_equalized#retorna a imagem equalizada

def joint_cumulative_histogram_equalization(imglows):#função de equalização de histograma conjunta
    all_imgs_combined = np.concatenate([img.flatten() for img in imglows])#concatena as imagens de baixa resolução
    hist, bins = np.histogram(all_imgs_combined, bins=256, range=[0, 256])#calcula o histograma das imagens concatenadas
    cumulative_distribution_function = hist.cumsum()#calcula a função de distribuição cumulativa
    cumulative_distribution_function_normalized = 255 * cumulative_distribution_function / cumulative_distribution_function[-1]#normaliza a função de distribuição cumulativa
    cumulative_distribution_function_m = np.ma.masked_equal(cumulative_distribution_function, 0)#mascara os valores iguais a 0
    cumulative_distribution_function_m = (cumulative_distribution_function_m - cumulative_distribution_function_m.min()) * 255 / (cumulative_distribution_function_m.max() - cumulative_distribution_function_m.min())#normaliza a função de distribuição cumulativa
    cumulative_distribution_function = np.ma.filled(cumulative_distribution_function_m, 0).astype('uint8')#preenche os valores da função de distribuição cumulativa e converte para uint8
    imglows_equalized = [np.reshape(cumulative_distribution_function_normalized[img.flatten()], img.shape) for img in imglows]#aplica a equalização de histograma para cada imagem de baixa resolução
    return imglows_equalized#retorna as imagens equalizadas

def gamma_correction(img, gamma): #função de correção gamma
    img_float = img.astype(np.float32) #converte para float o tipo da imagem
    corrected_img = np.clip(255 * (img_float / 255) ** (1 / gamma), 0, 255)#aplica a correção gamma
    return corrected_img.astype(np.float32)#retorna a imagem corrigida

def Superresolution(imglows):# chama a função de superesolução na qual realiza a interpolação das imagens de baixa resolução
    T = imglows[0].shape #pega o tamanho da imagem de baixa resolução
    imghighsuper = np.zeros((2 * T[0], 2 * T[1]), dtype=np.float32)#inicializa a imagem que será interpolada com zeros
    for i in range(T[0]):
        for j in range(T[1]):#percorre a imagem
            imghighsuper[2 * i, 2 * j] = imglows[0][i, j]#realiza a interpolação das 4 imagens em uma única
            imghighsuper[2 * i, 2 * j + 1] = imglows[1][i, j]
            imghighsuper[2 * i + 1, 2 * j] = imglows[2][i, j]
            imghighsuper[2 * i + 1, 2 * j + 1] = imglows[3][i, j]
    return imghighsuper#retorna a imagem interpolada

def main():
    imglow_base = str(input().rstrip())#lê a string do nome da imagem de baixa resolução e remove os espaços
    imghigh = str(input().rstrip())#lê a string do nome da imagem de alta resolução e remove os espaços
    F = int(input())#lê o valor de F(tipo de tratamento realizado na imagem)
    gamma = float(input())#lê o valor de gamma para o tratamento de correção gamma
    imglows = []#inicializa a lista de imagens de baixa resolução
    imghigh = imageio.imread(imghigh)#lê a imagem de alta resolução
    for i in range(4):
        filename = f"{imglow_base}{i}.png"#cria o nome da imagem de baixa resolução para realizar a busca 
        imglows.append(imageio.imread(filename)) #adiciona a imagem de baixa resolução na lista e lê ela
    if F == 0: #F==0 sem tratamento e aplica apenas superesolução
        imghighsuper = Superresolution(imglows) #chama a função de superesolução para o vetor de imagens de baixa resoluçao
    elif F == 1:#F==1 aplica equalização de histograma em cada imagem de baixa resolução e depois superesolução
        imglows = [single_image_histogram_equalization(img) for img in imglows] #chama a função de equalização de histograma para cada imagem de baixa resolução
        imghighsuper = Superresolution(imglows)#chama a função de superesolução para o vetor de imagens de baixa resoluçao
    elif F == 2:#F==2 aplica equalização de histograma conjunta e depois superesolução
        imglows = joint_cumulative_histogram_equalization(imglows)#chama a função de equalização de histograma conjunta para o vetor de imagens de baixa resoluçao
        imghighsuper = Superresolution(imglows)#chama a função de superesolução para o vetor de imagens de baixa resoluçao
    elif F == 3:#F==3 aplica correção gamma em cada imagem de baixa resolução e depois superesolução
        imglows = [gamma_correction(img, gamma) for img in imglows]#chama a função de correção gamma para cada imagem de baixa resolução
        imghighsuper = Superresolution(imglows)#chama a função de superesolução para o vetor de imagens de baixa resoluçao
    
    imghighsuper_normalized = np.clip(imghighsuper, 0, 255).astype(np.uint8) #normaliza a imagem interpolada para o tipo uint8 visto que a imagem tem 8 bytes e o valor máximo é 255
    imageio.imwrite("imghigh_modified.png", imghighsuper_normalized)#escreve a imagem interpolada em um arquivo
    imghighsuper = imageio.imread("imghigh_modified.png")#lê a imagem interpolada
    res = rmse(imghigh, imghighsuper)#chama a função de comparação entre imagem de refêrencia e imagem modificada
    print(f"{res:.4f}")#imprime o resultado da comparação

if __name__ == "__main__":
    main() #chama a função principal
