# INF7370 Apprentissage automatique 
# Travail pratique 3
# ===========================================================================

# #===========================================================================
# Ce modèle est un Autoencodeur Convolutif entrainé sur l'ensemble de données MNIST afin d'encoder et reconstruire les images des chiffres 2 et 7.
# MNIST est une base de données contenant des chiffres entre 0 et 9 Ècrits à la main en noire et blanc de taille 28x28 pixels
# Pour des fins d'illustration, nous avons pris seulement deux chiffres 2 et 7
#
# Données:
# ------------------------------------------------
# entrainement : classe '2': 1 000 images | classe '7': images 1 000 images
# validation   : classe '2':   200 images | classe '7': images   200 images
# test         : classe '2':   200 images | classe '7': images   200 images
# ------------------------------------------------

# >>> Ce code fonctionne sur MNIST.
# >>> Vous devez donc intervenir sur ce code afin de l'adapter aux données du TP3.
# >>> À cette fin repérer les section QUESTION et insérer votre code et modification à ces endroits

# ==========================================
# ======CHARGEMENT DES LIBRAIRIES===========
# ==========================================

# La libraire responsable du chargement des données dans la mémoire
#from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Le Model à compiler
from keras.models import Model

# Le type d'optimisateur utilisé dans notre modèle (RMSprop, adam, sgd, adaboost ...)
# L'optimisateur ajuste les poids de notre modèle par descente du gradient
# Chaque optimisateur a ses propres paramètres
# Note: Il faut tester plusieurs et ajuster les paramètres afin d'avoir les meilleurs résultats

from tensorflow.keras.optimizers import Adam

# Les types des couches utlilisées dans notre modèle
from keras.layers import Conv2D, MaxPooling2D, Input, BatchNormalization, UpSampling2D, Activation, Dropout, Flatten, \
    Dense

# Des outils pour suivre et gérer l'entrainement de notre modèle
from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping

# Configuration du GPU
import tensorflow as tf

# Affichage des graphes
import matplotlib.pyplot as plt

from keras import backend as K

# Afin de calculer le temps d'exécution
import time

# ==========================================
# ===============GPU SETUP==================
# ==========================================

# Configuration des GPUs et CPUs
config = tf.compat.v1.ConfigProto(device_count={'GPU': 2, 'CPU': 4})
sess = tf.compat.v1.Session(config=config)
#tf.compat.v1.keras.backend.set_session(sess);
tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)


# ==========================================
# ================VARIABLES=================
# ==========================================

# ******************************************************
#                       QUESTION DU TP
# ******************************************************
# 1) Ajuster les variables suivantes selon votre problème:
# - mainDataPath
# - training_ds_size
# - validation_ds_size
# - image_scale
# - image_channels
# - images_color_mode
# - fit_batch_size
# - fit_epochs
# ******************************************************

# Le dossier principal qui contient les données
mainDataPath = "donnees/"

# Le dossier contenant les images d'entrainement
trainPath = mainDataPath + "entrainement"

# Le dossier contenant les images de validation
validationPath = mainDataPath + "validation"

# Le nom du fichier du modèle à sauvegarder
model_path = "Model.keras"

# Le nombre d'images d'entrainement
# 80% de training et 20% de validations
training_ds_size = 2880  # total 2880 (1440 classe: requin et 1440 classe: dauphin)
validation_ds_size = 720  # total 720 (360 classe: requin et 360 classe: dauphin)


# Configuration des  images
image_scale = 128  # la taille des images
image_channels = 3  # le nombre de canaux de couleurs (1: pour les images noir et blanc; 3 pour les images en couleurs (rouge vert bleu) )
images_color_mode = "rgb"  # grayscale pour les image noir et blanc; rgb pour les images en couleurs
image_shape = (image_scale, image_scale,
               image_channels)  # la forme des images d'entrées, ce qui correspond à la couche d'entrée du réseau

# Configuration des paramètres d'entrainement
fit_batch_size = 32  # le nombre d'images entrainées ensemble: un batch
fit_epochs = 60  # Le nombre d'époques

# ==========================================
# ==================MODÈLE==================
# ==========================================

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#                       QUESTIONS DU TP
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# Ajuster les deux fonctions:
# 2) encoder
# 3) decoder
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# Couche d'entrée:
# Cette couche prend comme paramètre la forme des images (image_shape)
input_layer = Input(shape=image_shape)


# Partie d'encodage (qui extrait les features des images et les encode)
def encoder(input):

     # Block 1: 128x128 -> 64x64
    x = Conv2D(32, (3, 3), padding='same')(input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(32, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Dropout(0.10)(x)

    # Block 2: 64x64 -> 32x32
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Dropout(0.15)(x)

    # Block 3: 32x32 -> 16x16
    x = Conv2D(128, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(128, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    encoded = MaxPooling2D((2, 2), padding='same', name="embedding")(x)

    # encoded : La sortie de l'encodeur consistue l'embedding (ou les descripteurs extraites par l'encodeur)
    return encoded


# Partie de décodage (qui reconstruit les images à partir de leur embedding ou la sortie de l'encodeur)
def decoder(encoded):

   # Block 3': 16x16 -> 32x32
    x = Conv2D(128, (3, 3), padding='same')(encoded)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(128, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = UpSampling2D((2, 2))(x)

    # Block 2': 32x32 -> 64x64
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = UpSampling2D((2, 2))(x)

    # Block 1': 64x64 -> 128x128
    x = Conv2D(32, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(32, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = UpSampling2D((2, 2))(x)

    # la dernier couche doit avoir les mêmes dimensions de l'image d'entré (input)
    # (C'est très important, car nous somme entrain de reconstruire l'image d'entrée)
    x = Conv2D(image_channels, (3, 3), padding='same')(x)

    # la dernière couche doit passer par un sigmoide car les pixels des images sont
    # normalisées entre 0 et 1 et l'autoencodeur essaie de prédire chaque pixel par une valeur entre 0 et 1
    decoded = Activation('sigmoid')(x)
    return decoded


# Déclaration du modèle:
# La sortie de l'encodeur sert comme entrée à la partie decodeur
model = Model(input_layer, decoder(encoder(input_layer)))

# Affichage des paramétres du modèle
# Cette commande affiche un tableau avec les détails du modèle
# (nombre de couches et de paramétres ...)
model.summary()

# Compilation du modèle :
# loss: On définit la fonction de perte (généralement on utilise le MSE pour les autoencodeurs standards)
# optimizer: L'optimisateur utilisé avec ses paramétres (Exemple : optimizer=adam(learning_rate=0.001) )
# metrics: La valeur à afficher durant l'entrainement, metrics=['mse']
# On suit le loss (ou la difference) de l'autoencodeur entre les images d'entrée et les images de sortie
model.compile(loss='mse', optimizer=Adam(learning_rate=1e-3), metrics=['mse'])

# ==========================================
# ==========CHARGEMENT DES IMAGES===========
# ==========================================

# training_data_generator: charge les données d'entrainement en mémoire
# Les images sont normalisées (leurs pixels divisées par 255)
# Training: augmentation + normalisation
training_data_generator = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    width_shift_range=0.05,
    height_shift_range=0.05,
    zoom_range=0.10,
    horizontal_flip=True,
    fill_mode='nearest'
)

# validation_data_generator: charge les données de validation en mémoire
# Les images sont normalisées (leurs pixels divisées par 255)
validation_data_generator = ImageDataGenerator(rescale=1. / 255)

# training_generator: indique la méthode de chargement des données d'entrainement
training_generator = training_data_generator.flow_from_directory(
    trainPath, # Place des images d'entrainement
    color_mode =images_color_mode, # couleur des images
    target_size=(image_scale, image_scale),# taille des images
    batch_size = fit_batch_size, # MODIFICATION ICI ON NE CHARGE PAS TOUTES LES IMAGES D'UN COUP POUR EVITER UNE SURCHARGE DE LA RAM SUR COLAB
    class_mode ="input",
    shuffle=True #aide à la stabilisation
    )
    

# validation_generatory: indique la méthode de chargement des données de validation
validation_generator = validation_data_generator.flow_from_directory(
    validationPath, # Place des images d'entrainement
    color_mode =images_color_mode, # couleur des images
    target_size=(image_scale, image_scale),# taille des images
    batch_size = fit_batch_size, # MODIFICATION ICI ON NE CHARGE PAS TOUTES LES IMAGES D'UN COUP POUR EVITER UNE SURCHARGE DE LA RAM SUR COLAB
    class_mode ="input",
    shuffle=False) # Comme nous somme entrain de reconstruire les images, alors
                        # la classe de chacune des pixels de sorite est le pixel d'entrée elle même(Input pixel)

# ==========================================
# ==============ENTRAINEMENT================
# ==========================================

steps_per_epoch = training_ds_size // fit_batch_size
validation_steps = validation_ds_size // fit_batch_size

print("Train batches per epoch:", training_ds_size // fit_batch_size)
print("Val batches per epoch:", validation_ds_size // fit_batch_size)
print("Class indices:", training_generator.class_indices)

# Savegarder le modèle avec le minimum loss sur les données de validation (monitor='val_loss')
# Note: on sauvegarde le modèle seulement quand le validation loss (la perte) diminue
# le loss ici est la difference entre les images originales (input) et les images reconstruites (output)
modelcheckpoint = ModelCheckpoint(filepath=model_path,
                                  monitor='val_loss', verbose=1, save_best_only=True, mode='auto')

# EarlyStop pour éviter le surraprentissage, utilisation d'une patience modéré de 8
earlystop = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)

# On démarre le temps d'entraînement lorsqu'on entraîne le modèle (pas avant pour éviter de mesurer les autres instructions)
start_time = time.time()

# entrainement du modèle
# On remarque que pour la fonction "fit", la valeur de "x" (données d'entrainement) et celles de "y" (étiquettes) sont les mêmes
# C'est parce qu'on est entrain de reconstruire les pixels de l'image d'entrée
autoencoder = model.fit(
    training_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=fit_epochs,
    verbose=1,
    callbacks=[modelcheckpoint, earlystop],
    shuffle=False,
    validation_data=validation_generator,
    validation_steps=validation_steps
)

# ==========================================
# ========AFFICHAGE DES RESULTATS===========
# ==========================================

# ***********************************************
#                    QUESTION
# ***********************************************
#
# 4) Afficher le temps d'execution
#
# ***********************************************

end_time = time.time()
execution_time_minutes = (end_time - start_time) / 60.0
print(f"\nTemps total d'exécution (entrainement) : {execution_time_minutes:.2f} minutes")

# ***********************************************
#                    QUESTION
# ***********************************************
#
# 5) Afficher la courbe de  perte par époque (loss over epochs)
#
# ***********************************************

loss = autoencoder.history['loss']
val_loss = autoencoder.history['val_loss']
epochs_range = range(1, len(loss) + 1)

plt.figure(figsize=(8, 5))
plt.plot(epochs_range, loss, label='Training loss')
plt.plot(epochs_range, val_loss, label='Validation loss')
plt.title('Loss over epochs')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("loss_curve.png", dpi=200)
plt.show()
plt.close()

print("Figure sauvegardée : loss_curve.png")

print(f"Min training loss: {min(loss):.6f}")
print(f"Min validation loss: {min(val_loss):.6f}")
print(f"Best epoch (min val_loss): {val_loss.index(min(val_loss)) + 1}")
