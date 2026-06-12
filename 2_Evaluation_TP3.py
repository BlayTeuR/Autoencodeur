# **************************************************************************
# INF7370 Apprentissage automatique 
# Travail pratique 3
# ===========================================================================

#===========================================================================
# Dans ce script, on évalue l'autoencodeur entrainé dans 1_Modele.py sur les données tests.
# On charge le modèle en mémoire puis on charge les images tests en mémoire
# 1) On évalue la qualité des images reconstruites par l'autoencodeur
# 2) On évalue avec une tache de classification la qualité de l'embedding
# 3) On visualise l'embedding en 2 dimensions avec un scatter plot


# ==========================================
# ======CHARGEMENT DES LIBRAIRIES===========
# ==========================================

# La libraire responsable du chargement des données dans la mémoire
#from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Affichage des graphes et des images
import matplotlib.pyplot as plt

# La librairie numpy
import numpy as np

# Configuration du GPU
import tensorflow as tf

# Utlilisé pour charger le modèle
from keras.models import load_model
from keras import Model

# Utilisé pour normaliser l'embedding
from sklearn.preprocessing import StandardScaler

from keras import backend as K

from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import GridSearchCV

# ==========================================
# ===============GPU SETUP==================
# ==========================================

# Configuration des GPUs et CPUs
config = tf.compat.v1.ConfigProto(device_count={'GPU': 2, 'CPU': 4})
sess = tf.compat.v1.Session(config=config)
#tf.compat.v1.keras.backend.set_session(sess);
tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)



# ==========================================
# ==================MODÈLE==================
# ==========================================

# Chargement du modéle (autoencodeur) sauvegardé dans la section 1 via 1_Modele.py
model_path = "Model.keras"
autoencoder = load_model(model_path)

# ==========================================
# ================VARIABLES=================
# ==========================================

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#                       QUESTIONS
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# 1) A ajuster les variables suivantes selon votre problème:
# - mainDataPath
# - number_images
# - number_images_class_x
# - image_scale
# - images_color_mode
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>


# L'emplacement des images
mainDataPath = "donnees/"

# On évalue le modèle sur les images tests
datapath = mainDataPath + "test"

# Le nombre des images de test à évaluer
number_images = 600 # 400 images
number_images_class_0 = 300 # 200 images pour la classe de la classe requin
number_images_class_1 = 300 # 200 images pour la classe de la classe dauphin

# Les étiquettes (classes) des images
labels = np.array([0] * number_images_class_0 +
                  [1] * number_images_class_1)

# La taille des images
image_scale = 128 # On utilise la même taille que celle utilisé pour les images d'entraînement

# La couleur des images
images_color_mode = "rgb"  # grayscale ou rgb

# ==========================================
# =========CHARGEMENT DES IMAGES============
# ==========================================

# Chargement des images test
data_generator = ImageDataGenerator(rescale=1. / 255)

generator = data_generator.flow_from_directory(
    datapath, # Place des images d'entrainement
    color_mode=images_color_mode, # couleur des images
    target_size=(image_scale, image_scale),# taille des images
    batch_size= number_images, # nombre d'images total à charger en mémoire
    class_mode=None,
    shuffle=False) # pas besoin de bouleverser les images

x = generator.__next__()

# ***********************************************
#                  QUESTIONS
# ***********************************************
#
# 2) Reconstruire les images tests en utilisant l'autoencodeur entrainé dans la première étape.
# Pour chacune des classes: Afficher une image originale ainsi que sa reconstruction.
# Afficher le titre de chaque classe au-dessus de l'image
# Note: Les images sont normalisées (entre 0 et 1), alors il faut les multiplier
# par 255 pour récupérer les couleurs des pixels
#
# ***********************************************

# 1) Reconstruire toutes les images de test
x_recon = autoencoder.predict(x, verbose=0)

# 2) Récupérer les noms des classes + l'index des classes par image
# (flow_from_directory crée class_indices et classes même si class_mode=None)
class_indices = generator.class_indices
index_to_class = {v: k for k, v in class_indices.items()}
y = generator.classes

# 3) Prendre 1 image représentative par classe (la première occurrence)
unique_classes = sorted(list(index_to_class.keys()))
picked_idx = []
for c in unique_classes:
    idx_c = np.where(y == c)[0][0]
    picked_idx.append(idx_c)

# 4) Affichage: 2 lignes (Original / Reconstruit), 1 colonne par classe
n_classes = len(unique_classes)
plt.figure(figsize=(4 * n_classes, 6))

for col, idx in enumerate(picked_idx):
    class_name = index_to_class[unique_classes[col]]

    # Originale + reconstruction
    img_o = x[idx]
    img_r = x_recon[idx]

    # Passage en uint8 (consigne: images normalisées -> *255) + gestion gray/rgb
    img_o_255 = (img_o * 255).astype(np.uint8)
    img_r_255 = (img_r * 255).astype(np.uint8)

    # --- Original ---
    ax1 = plt.subplot(2, n_classes, col + 1)
    if img_o_255.shape[-1] == 1:
        plt.imshow(img_o_255.squeeze(), cmap="gray")
    else:
        plt.imshow(img_o_255)
    plt.title(f"{class_name} - Original")
    plt.axis("off")

    # --- Reconstruite ---
    ax2 = plt.subplot(2, n_classes, n_classes + col + 1)
    if img_r_255.shape[-1] == 1:
        plt.imshow(img_r_255.squeeze(), cmap="gray")
    else:
        plt.imshow(img_r_255)
    plt.title(f"{class_name} - Reconstruit")
    plt.axis("off")

    # plt.imsave(f"original_{class_name}.png", img_o_255.squeeze(), cmap="gray")
    # plt.imsave(f"reconstruit_{class_name}.png", img_r_255.squeeze(), cmap="gray")

plt.tight_layout()
plt.show()

# ***********************************************
#                  QUESTIONS
# ***********************************************
#
# 3) Définire un modèle "encoder" qui est formé de la partie encodeur de l'autoencodeur
# Appliquer ce modèle sur les images afin de récupérer l'embedding
# Note: Il est "nécessaire" d'appliquer la fonction (flatten) sur l'embedding
# afin de réduire la représentation de chaque image en un seul vecteur
#
# ***********************************************

# Défintion du modèle
input_layer_index = 0 # l'indice de la première couche de l'encodeur (input)
output_layer_index = 6 # l'indice de la dernière couche (la sortie) de l'encodeur (dépend de votre architecture)
# note: Pour identifier l'indice de la dernière couche de la partie encodeur, vous pouvez utiliser la fonction "model.summary()"
# chaque ligne dans le tableau affiché par "model.summary" est compté comme une couche

#encoder = Model(autoencoder.layers[input_layer_index].input, autoencoder.layers[output_layer_index].output)
encoder = Model(autoencoder.input, autoencoder.layers[output_layer_index].output)

# Appliquer l'encoder sur les images tests => embedding
embedding = encoder.predict(x, verbose=0)

print("Shape embedding (avant flatten) :", embedding.shape)

# Flatten: chaque image -> 1 vecteur
embedding_flat = embedding.reshape(embedding.shape[0], -1)

print("Shape embedding_flat :", embedding_flat.shape)


# ***********************************************
#                  QUESTIONS
# ***********************************************
#
# 4) Normaliser le flattened embedding (les vecteurs recupérés dans question 3)
# en utilisant le StandardScaler
# ***********************************************

scaler = StandardScaler()

# Normalisation (fit + transform sur le jeu de test)
embedding_flat_norm = scaler.fit_transform(embedding_flat)

print("Shape embedding normalisé :", embedding_flat_norm.shape)
print("Moyenne (≈0) :", np.mean(embedding_flat_norm))
print("Écart-type (≈1) :", np.std(embedding_flat_norm))

# ***********************************************
#                  QUESTIONS
# ***********************************************
#
# 5) Appliquer un SVM Linéaire sur les images originales (avant l'encodage par le modèle)
# Entrainer le modèle avec le cross-validation
# Afficher la métrique suivante :
#    - Accuracy
# ***********************************************

# X_original: flatten des images (pixels)
X_original = x.reshape(x.shape[0], -1)
y_labels = y

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

pipe_pixels = Pipeline([
    ("scaler", StandardScaler(with_mean=True, with_std=True)),
    ("svm", LinearSVC(max_iter=20000, random_state=42))
])

param_grid_pixels = {
    "svm__C": [0.001, 0.01, 0.1, 1, 10, 100],
    "svm__class_weight": [None, "balanced"]
}

grid_pixels = GridSearchCV(
    estimator=pipe_pixels,
    param_grid=param_grid_pixels,
    scoring="accuracy",
    cv=cv,
    n_jobs=-1,
    verbose=2
)

grid_pixels.fit(X_original, y_labels)

print("\n====== MEILLEURS PARAMÈTRES (Pixels) ======")
print(grid_pixels.best_params_)
print(f"Meilleure accuracy CV (pixels): {grid_pixels.best_score_:.4f}")

# ***********************************************
#                  QUESTIONS
# ***********************************************
#
# 6) Appliquer un SVC Linéaire sur le flattened embedding normalisé
# Entrainer le modèle avec le cross-validation
# Afficher la métrique suivante :
#    - Accuracy
# ***********************************************


cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

pipe_embed = Pipeline([
    ("scaler", StandardScaler(with_mean=True, with_std=True)),
    ("pca", PCA(random_state=42)),
    ("svm", LinearSVC(max_iter=20000, random_state=42))
])

param_grid = {
    "pca__n_components": [64, 128, 256, 512],
    "svm__C": [0.01, 0.1, 1, 10, 100],
    "svm__class_weight": [None, "balanced"]
}

# Optimisation: GridSearch sur PCA dim + C
grid = GridSearchCV(
    estimator=pipe_embed,
    param_grid=param_grid,
    scoring="accuracy",
    cv=cv,
    n_jobs=-1,
    verbose=2
)

grid.fit(embedding_flat, y_labels)

print("\n====== MEILLEURS PARAMÈTRES (Embedding) ======")
print(grid.best_params_)
print(f"Meilleure accuracy CV: {grid.best_score_:.4f}")

# ***********************************************
#                  QUESTIONS
# ***********************************************
#
# 7) Appliquer TSNE sur le flattened embedding afin de réduire sa dimensionnalité en 2 dimensions
# Puis afficher les 2D features dans un scatter plot en utilisant 2 couleurs(une couleur par classe)
# ***********************************************

# Pré-réduction PCA (accélère t-SNE)
pca = PCA(n_components=50, random_state=42)
embedding_for_tsne = pca.fit_transform(embedding_flat_norm)

# t-SNE vers 2 dimensions
tsne = TSNE(
    n_components=2,
    perplexity=40,
    n_iter=2000,
    learning_rate="auto",
    init="pca",
    random_state=42
)

embedding_2d = tsne.fit_transform(embedding_for_tsne)

# Scatter plot: 2 couleurs (une par classe)
plt.figure(figsize=(8, 6))
plt.scatter(embedding_2d[y_labels == 0, 0], embedding_2d[y_labels == 0, 1], alpha=0.7, label="Classe 0")
plt.scatter(embedding_2d[y_labels == 1, 0], embedding_2d[y_labels == 1, 1], alpha=0.7, label="Classe 1")

plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.title("t-SNE sur le flattened embedding (PCA 50 → t-SNE)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("tsne_embedding_scatter.png", dpi=300, bbox_inches="tight")
plt.show()
