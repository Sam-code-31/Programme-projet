import cv2
import numpy as np
from time import time
import multiprocessing

class SAM :
    
    def __init__(self,image1,image2,image3):
        self.image1 = image1
        self.image2 = image2
        self.image3 = image3

    def debruitage(self,image):

        ker = 3
        image_modif = image.copy()
        image_aggrandie = np.pad(image, pad_width=ker// 2  , mode='constant', constant_values=255)
        pixel_traite = 0
        aire = image.shape[0]*image.shape[1]
        pourcentage = 0
        for x in range(image.shape[0]):
            for y in range(image.shape[1]):
                carree = image_aggrandie[x:x+ker, y:y+ker]
                image_modif[x, y] = np.median(carree)
                
                pixel_traite +=1
                if round((pixel_traite/aire) *100) >= pourcentage :
                    print('le traitement est a {} %'.format(pourcentage))
                    pourcentage += 10
        return image_modif.astype(np.uint8)

    def debruitage_moy(self,image):

        ker = 3
        image_modif = image.copy()
        image_aggrandie = np.pad(image, pad_width=ker// 2  , mode='constant', constant_values=255)
        pixel_traite = 0
        aire = image.shape[0]*image.shape[1]
        pourcentage = 0
        for x in range(image.shape[0]):
            for y in range(image.shape[1]):
                carree = image_aggrandie[x:x+ker, y:y+ker]
                image_modif[x, y] = np.mean(carree)
                
                pixel_traite +=1
                if round((pixel_traite/aire) *100) >= pourcentage :
                    print('le traitement est a {} %'.format(pourcentage))
                    pourcentage += 10
        return image_modif.astype(np.uint8)

    def debruitage_gaussien(self, image):
        ker = 12
        sigma = 2
        kernel = np.zeros((ker, ker))
        center = ker // 2
        total = 0  # Pour la normalisation
        image_modif = np.zeros_like(image)
        pixel_traite = 0
        aire = image.shape[0]*image.shape[1]
        pourcentage = 0
        
        for i in range(ker):
            for j in range(ker):
                x = i - center
                y = j - center
                kernel[i, j] = np.exp(-(x**2 + y**2) / (2 * sigma**2))
                total += kernel[i, j]
        
        # Normalisation pour que la somme des coefficients = 1
        kernel /= total
        image_aggrandie = np.pad(image, pad_width=ker// 2  , mode='constant', constant_values=255)
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                region = image_aggrandie[i:i+ker, j:j+ker]
                image_modif[i, j] = np.sum(region * kernel)

                pixel_traite +=1
                if round((pixel_traite/aire) *100) >= pourcentage :
                    print('le traitement est a {} %'.format(pourcentage))
                    pourcentage += 10
        return image_modif.astype(np.uint8)


def afficher_image(image):
    '''
    affiche l'image grâce a son nom et applique le debruitage median
    '''
    début = time()
    B, V, R = cv2.split(image)
    image_brutee = SAM(B, V, R)
    choix = int(input('quel méthode \n gaussien : (1) \n median : (2) \n moy : (3) \n'))
    if choix  == 2: 
        # Utilisation de multiprocessing Pool
        with multiprocessing.Pool(processes=3) as pool:
            Bleu, Vert, Rouge = pool.map(image_brutee.debruitage, [image_brutee.image1, image_brutee.image2, image_brutee.image3])
    elif choix == 1  : 
        # Utilisation de multiprocessing Pool
        with multiprocessing.Pool(processes=3) as pool:
            Bleu, Vert, Rouge = pool.map(image_brutee.debruitage_gaussien, [image_brutee.image1, image_brutee.image2, image_brutee.image3])
    elif choix == 3  : 
        # Utilisation de multiprocessing Pool
        with multiprocessing.Pool(processes=3) as pool:
            Bleu, Vert, Rouge = pool.map(image_brutee.debruitage_moy, [image_brutee.image1, image_brutee.image2, image_brutee.image3])

    Image_finale = cv2.merge([Bleu, Vert, Rouge])
    fin = time()
    return fin - début, Image_finale

    


if __name__ == '__main__':
    image_brute = cv2.imread(input("quel nom ? ")+'.png')
    durée, image_brute = afficher_image(image_brute)
    print('le programme a mit {} secondes'.format(durée))
    cv2.imwrite(input("quel nom ? ")+".png", image_brute)


