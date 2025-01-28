import numpy as np
import imageio.v3 as imageio
import math
def rmse(ref_img, filt_img):#função para calcular a diferença entre duas imagens
    N=ref_img.shape #pega o tamanho da imagem
    soma = 0.00 #inicializa a soma
    ref_img = ref_img.astype(np.float64) #converte a imagem de referência para float
    filt_img = filt_img.astype(np.float64)
    for i in range(N[0]):
        for j in range(N[1]):
            erro_quadrado = (ref_img[i, j] - filt_img[i, j]) ** 2#calcula o erro quadrado
            soma += erro_quadrado#soma o erro quadrado
    rmse = np.sqrt(soma / (N[0] * N[1]))#tira a raiz e calcula o rmse
    return rmse # retorna resultado
def normalizar(img_filtrada):#função para normalizar a imagem filtrada
    img_real = img_filtrada.real #pega a parte real da imagem
    img_normalizada = (img_real - np.min(img_real)) / (np.max(img_real) - np.min(img_real))#normaliza a imagem
    img_uint8 = (img_normalizada * 255).astype(np.uint8)#converte a imagem para uint8
    return img_uint8 #retorna a imagem normalizada
def fourier_transform(img):#função para calcular a transformada de fourier
    img_fft = np.fft.fft2(img)#calcula a transformada de fourier
    img_fft_shifted = np.fft.fftshift(img_fft)#shifta a transformada
    return img_fft_shifted #retorna a transformada
def fourier_inverse(img,filtro):#função para calcular a transformada inversa de fourier e o filtro
    filtered_fft_shifted = img * filtro#aplica o filtro na transformada
    img_fft = np.fft.ifftshift(filtered_fft_shifted)#shifta a transformada
    img_ifft = np.fft.ifft2(img_fft)#calcula a transformada inversa
    return img_ifft #retorna a imagem inversa
def low_pass_filter(img, r): #função para aplicar o filtro low-pass
    img_fft_shifted = fourier_transform(img) #calcula a transformada de fourier
    M, N = img.shape #pega o tamanho da imagem
    center_x, center_y = M // 2, N // 2 #calcula o centro da imagem
    low_pass = np.zeros_like(img_fft_shifted)  
    for x in range(M):
        for y in range(N):
            if np.sqrt((x - center_x)**2 + (y - center_y)**2) <= r: #aplica o filtro low-pass
                low_pass[x, y] = 1
            else:
                low_pass[x, y] = 0
    img_filtrada = fourier_inverse(img_fft_shifted,low_pass) #calcula a transformada inversa e o filtro
    img_normalizada=  normalizar(img_filtrada)
    imageio.imwrite("ifftd2.png", img_normalizada) #salva a imagem resultante
    return img_normalizada
def high_pass_filter(img, r): #função para aplicar o filtro high-pass
    img_fft_shifted = fourier_transform(img)
    M, N = img.shape
    center_x, center_y = M // 2, N // 2
    high_pass = np.zeros_like(img_fft_shifted)  # Assegura que corresponda à forma do array complexo
    for x in range(M):
        for y in range(N):
            if np.sqrt((x - center_x)**2 + (y - center_y)**2) <= r: #aplica o filtro high-pass
                high_pass[x, y] = 0
            else:
                high_pass[x, y] = 1 # Isso indica alta passagem
    img_filtrada = fourier_inverse(img_fft_shifted,high_pass)
    img_normalizada= normalizar(img_filtrada)
    imageio.imwrite("ifftd2.png", img_normalizada) #salva a imagem resultante
    return img_normalizada
def band_stop_filter(img, r0, r1):#função para aplicar o filtro band-stop
    img_fft_shifted = fourier_transform(img) #calcula a transformada de fourier
    M, N = img.shape
    center_x, center_y = M // 2, N // 2
    band_stop_filter = np.zeros_like(img_fft_shifted)  # Ensure it matches the complex array shape
    for x in range(M):
        for y in range(N):
            if np.sqrt((x - center_x)**2 + (y - center_y)**2) <= r0 and np.sqrt((x - center_x)**2 + (y - center_y)**2) >= r1: #aplica o filtro band-stop
                band_stop_filter[x, y] = 0
            else:
                band_stop_filter[x, y] = 1
    img_filtrada = fourier_inverse(img_fft_shifted,band_stop_filter)
    img_normalizada= normalizar(img_filtrada)
    imageio.imwrite("ifftd2.png", img_normalizada) #salva a imagem resultante
    return img_normalizada
def laplacian_high_pass(img):#função para aplicar o filtro laplacian high-pass
    img_fft_shifted = fourier_transform(img)
    M, N = img.shape
    center_x, center_y = M // 2, N // 2
    laplacian_high_pass = np.zeros_like(img_fft_shifted)  # Ensure it matches the complex array shape
    for x in range(M):
        for y in range(N):
            sum=np.sqrt((x - center_x)**2 + (y - center_y)**2)
            laplacian_high_pass[x, y] = -4 * np.pi**2 * sum**2
    img_filtrada = fourier_inverse(img_fft_shifted,laplacian_high_pass)#calcula a transformada inversa e o filtro
    img_normalizar= normalizar(img_filtrada)
    imageio.imwrite("ifftd2.png", img_normalizar) #salva a imagem resultante
    return img_normalizar
def gaussian_low_pass(img, sigma1, sigma2): #função para aplicar o filtro gaussian low-pass
    img_fft_shifted = fourier_transform(img) #calcula a transformada de fourier
    M, N = img.shape
    center_x, center_y = M // 2, N // 2
    gaussian_low_pass = np.zeros_like(img_fft_shifted) 
    for x in range(M):
        for y in range(N):
            x1=(x-center_x)**2/(2*sigma1**2)+(y-center_y)**2/(2*sigma2**2) #aplica o filtro gaussian low-pass
            gaussian_low_pass[x, y] = math.exp(-x1)
    img_filtrada = fourier_inverse(img_fft_shifted,gaussian_low_pass) #calcula a transformada inversa e o filtro
    img_normalizada= normalizar(img_filtrada)
    imageio.imwrite("ifftd2.png", img_normalizada) #salva a imagem resultante
    return img_normalizada
def main():
    input_img = str(input().rstrip()) # lê o nome da imagem de input
    ref_img = str(input().rstrip()) # lê o nome da imagem de referência
    i = int(input()) # lê o valor de i (tipo de filtro realizado na imagem)
    img = imageio.imread(input_img) # lê a imagem de input
    ref = imageio.imread(ref_img) # lê a imagem de referência
    if i == 0: # filtro low-pass com raio r
        r = int(input())
        img_my= low_pass_filter(img, r)
        print(f"{rmse(ref, img_my):.4f}") 
    elif i == 1: # filtro high-pass com raio r
        r = int(input())
        img_my= high_pass_filter(img, r)
        print(f"{rmse(ref, img_my):.4f}")
    elif i == 2: # filtro Band-stop com raio r0 e r1
        r0 = int(input())
        r1 = int(input())
        img_my= band_stop_filter(img, r0, r1)
        print(f"{rmse(ref, img_my):.4f}")
    elif i == 3: # laplacian high-pass
        img_my= laplacian_high_pass(img)
        print(f"{rmse(ref, img_my):.4f}")
    elif i == 4: # gaussian low-pass com sigma1 e sigma2
        sigma1 = int(input())
        sigma2 = int(input())
        img_my= gaussian_low_pass(img, sigma1, sigma2)
        print(f"{rmse(ref, img_my):.4f}")

if __name__ == "__main__":
    main() #chama a função principal 
