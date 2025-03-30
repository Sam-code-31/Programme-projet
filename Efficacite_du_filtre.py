import numpy as np
import cv2
import csv
from math import *



def Frobenius(Matrice1, Matrice2):
    '''
    Utilise la méthode Manhattan pour calculer la distance entre 2 matrice de dimension 2, 
    c'est une méthode rapide mais couteuse car utilise la racine carrée mais plus précise 
    : param : 2 matrice de taille identique 
    : return : la distance entre les 2 matrices 
    '''
    return np.sqrt(np.sum((Matrice1 - Matrice2) ** 2))

def EMQ(Matrice1, Matrice2):
    '''
    Calcule l'erreur moyenne quadratique entre les images en comparant l'erreur pixel par pixel
    Moins le chiffre est élévé plus le filtre est efficace
    '''
    return np.sum((Matrice1 - Matrice2)**2)/(Matrice1.shape[0]*Matrice1.shape[1])

def PSNR (Matrice1, Matrice2):
    '''
    Calcule le Rapport signal/bruit de crête.
    Plus le chiffre est élévé plus le filtre est efficace
    '''
    Max_pixel = 255
    return 10 * log10((Max_pixel)**2 / EMQ(Matrice1, Matrice2))


def calcul_de_distance_Frobenius(Image1, Image2):
    '''
    Calcule la distance entre deux image couleur
    : param : les deux images ous forme de tableau 3D numpy
    : return : la distance
    '''
    MB_Bleu, MB_Vert, MB_Rouge =  cv2.split(Image1)
    MD_Bleu, MD_Vert, MD_Rouge =  cv2.split(Image2)
    D1 = Frobenius(MB_Bleu, MD_Bleu)
    D2 = Frobenius(MB_Rouge, MD_Rouge)
    D3 = Frobenius(MB_Vert,MD_Vert)
    return ((D1 + D2 + D3) /3)

def calcul_de_distance_EMQ(Image1, Image2):
    '''
    Calcule la distance entre deux image couleur
    : param : les deux images ous forme de tableau 3D numpy
    : return : la distance
    '''
    MB_Bleu, MB_Vert, MB_Rouge =  cv2.split(Image1)
    MD_Bleu, MD_Vert, MD_Rouge =  cv2.split(Image2)
    D1 = EMQ(MB_Bleu, MD_Bleu)
    D2 = EMQ(MB_Rouge, MD_Rouge)
    D3 = EMQ(MB_Vert,MD_Vert)
    return ((D1 + D2 + D3) /3)


def calcul_de_distance_PSNR(Image1, Image2):
    '''
    Calcule la distance entre deux image couleur
    : param : les deux images ous forme de tableau 3D numpy
    : return : la distance
    '''
    MB_Bleu, MB_Vert, MB_Rouge =  cv2.split(Image1)
    MD_Bleu, MD_Vert, MD_Rouge =  cv2.split(Image2)
    D1 = PSNR(MB_Bleu, MD_Bleu)
    D2 = PSNR(MB_Rouge, MD_Rouge)
    D3 = PSNR(MB_Vert,MD_Vert)
    return ((D1 + D2 + D3) /3)

if __name__ == '__main__' :
    image_bruitee = cv2.imread(input("Nom de l'image au format png : ")+".png")
    image_debruitee = cv2.imread("Nom de l'image 2 au format png .png")
    print(f'ceci est la distence de Frobenius : {calcul_de_distance_Frobenius(image_bruitee,image_debruitee)}')
    print(f'ceci est la distence de PSNR : {calcul_de_distance_PSNR(image_bruitee,image_debruitee)}')
    print(f'ceci est la distence de EMQ : {calcul_de_distance_EMQ(image_bruitee,image_debruitee)}')