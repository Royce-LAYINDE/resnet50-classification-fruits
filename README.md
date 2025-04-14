# Fine-Tuning ResNet50 pour la Classification d’Images de Fruits

Ce projet met en œuvre le fine-tuning du modèle ResNet-50 pré-entraîné pour la classification d’images de fruits. Il utilise le dataset "fruits-360" disponible sur Hugging Face, et propose un modèle performant, prêt à être déployé sous forme d’API Flask.

## Objectifs

- Adapter un modèle de deep learning pré-entraîné (ResNet-50) à un dataset personnalisé.
- Obtenir un classificateur performant pour 131 catégories de fruits.
- Déployer le modèle dans une application accessible via une API.

## Données

- **Source** : [PedroSampaio/fruits-360](https://huggingface.co/datasets/PedroSampaio/fruits-360)
- **Contenu** : 131 classes de fruits, images de différentes tailles
- **Prétraitement** :
  - Redimensionnement aléatoire (`RandomResizedCrop`)
  - Conversion en tenseurs PyTorch (`ToTensor`)
  - Normalisation avec les paramètres de ResNet-50

## Modèle utilisé

- **Architecture** : ResNet-50 (pré-entraîné)
- **Bibliothèque** : `transformers` de Hugging Face
- **Fine-tuning** :
  - 10 époques
  - Batch size : 16
  - Learning rate : 5e-5
  - Meilleur modèle sauvegardé selon la précision (accuracy)

## Résultats

- **Accuracy sur l’ensemble de test** : 98.5 %
- **Évaluation** : matrice de confusion montrant de bonnes performances, même sur des fruits similaires

## Déploiement

- Le modèle a été exporté pour être utilisé via une API Flask :
  - `pytorch_model.bin` : poids du modèle
  - `preprocessor_config.json` : configuration du préprocesseur
  - `id_to_label.json` : correspondance ID → nom du fruit
- Un script Python (`main.py`) permet de charger le modèle et prédire sur de nouvelles images

## Utilisation

1. Cloner le dépôt :
   ```bash
   git clone https://github.com/Royce-LAYINDE/resnet50-classification-fruits.git
   cd resnet50-classification-fruits
   ```
2. Installer les dépendances :

  ```bash
  pip install -r requirements.txt
  ```
3. Lancer le serveur Flask :

```bash
python main.py
```
4. Envoyer une image à l’API pour obtenir la prédiction.

## Structure du projet
- main.py : API Flask pour tester le modèle
- mymodel/ : modèle entraîné, config et labels
- notebooks/ : notebook d’entraînement

## Auteur
Projet réalisé par [Malick Royce LAYINDE](https://roylab.xyz/) dans le cadre d’un exercice de fine-tuning de modèles de vision par ordinateur.
