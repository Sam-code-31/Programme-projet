import cv2
import numpy as np

'''
possible erreur 3 bruitage different sur chaque image de couleur
a discuté avec les autres
'''

def Bruitage_poisson(image,pourcentage):
    image_modif = image.copy()
    taille = image.shape[0] * image.shape[1]
    for a in range(round(taille * pourcentage)):
        ligne, colonne = np.random.randint(0,image.shape[0]), np.random.randint(0,image.shape[1])
        image_modif[ligne][colonne] = np.random.randint(0, 255)
        
    return image_modif


def Separateur_de_Couleur(image):
    '''
    on lui donne une image(un tableau numpy a 3D) et on le divise en 3 tableau de couleur verte, bleu et rouge
    '''
    return cv2.split(image)       


if __name__ == '__main__':
    image_finale =[]
    image = cv2.imread(input('Nom : ') + '.png')  #mes l'image en noir et blanc
    Couleurs = Separateur_de_Couleur(image)
    pourcentage = int(input('Combien ? '))/100
    for Couleur in Couleurs:
         image_finale.append(Bruitage_poisson(Couleur, pourcentage))
    image = cv2.merge(image_finale)
    cv2.imshow('slayyy', image)
    cv2.waitKey(0)
    cv2.imwrite("image_Buitee_couleur_{}%.png".format(pourcentage*100), image)
    



