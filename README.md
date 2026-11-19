# Déploiement Fashion-MNIST avec FastAPI

Mustapha Aarab et Aya Agrigah

## Interface Streamlit

L'interface française propose 20 exemples Fashion-MNIST inclus dans le dépôt,
l'import PNG/JPEG, un aperçu des pixels envoyés, la classe prédite, la confiance,
la latence, le téléchargement JSON et les résultats de la campagne du TP.
Le rapport Word est disponible dans l'onglet À propos.

Après installation des dépendances, lancer FastAPI dans un premier terminal :

```powershell
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

Puis lancer l'interface dans un deuxième terminal :

```powershell
python -m streamlit run streamlit_app.py
```

Ouvrir http://localhost:8501. L'interface appelle réellement l'API par HTTP.
La variable d'environnement `FASHION_API_URL` permet d'utiliser un autre serveur.
Pour héberger l'interface à distance, déployer aussi le serveur FastAPI et configurer
cette variable avec son URL accessible ; publier le dépôt GitHub seul ne démarre
pas les services. Les imports sont limités à 5 Mo et convertis en niveaux de gris
28 × 28 ; utiliser un vêtement isolé sur fond noir. Les photos ordinaires ne suivent
pas nécessairement la distribution Fashion-MNIST.

Les exemples proviennent du jeu de test Fashion-MNIST de Zalando Research :
https://github.com/zalandoresearch/fashion-mnist (licence MIT).

## Installation et exécution

Utiliser Python 3.12. Conserver `fashion_mnist.keras` dans le même dossier que `app.py`.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

Dans un second terminal, après activation du même environnement :

```powershell
python test_api.py --count 20 --show
```

Le premier lancement télécharge Fashion-MNIST dans `.keras/`, local au projet.
Une connexion Internet est nécessaire pour ce téléchargement et l'installation.
`--show` ouvre les figures ; sans cette option, elles sont enregistrées sans ouvrir de fenêtre.
Swagger est accessible à http://127.0.0.1:8000/docs et l'état à http://127.0.0.1:8000/health.

## Résultats

- `resultats/resultats.json` : réponses réelles, statistiques, métadonnées et contrôles.
- `resultats/predictions.csv` : tableau des prédictions.
- `resultats/image_0_payload.json` : requête complète à copier dans Swagger.
- `captures/` : captures Swagger et figures des images testées.
- Le rapport Word contient l'analyse et les réponses aux dix questions du sujet.

Le script teste les 20 premières images après une requête d'échauffement exclue
des statistiques. La latence renvoyée par l'API couvre le modèle, softmax et la
conversion en NumPy ; la durée HTTP mesurée côté client est enregistrée séparément.
Les chiffres de latence dépendent de la machine et varient d'une exécution à l'autre.
Les captures Swagger peuvent afficher une latence différente du tableau, car il
s'agit de requêtes supplémentaires réalisées pour les captures.

Le test avec trois pixels doit renvoyer HTTP 400 et un message explicite.
Les assertions contrôlent aussi la taille 785, les bornes des pixels, les champs
manquants, un type incorrect et l'équivalence brut/normalisé. Une assertion en échec
arrête le script avec un code non nul.
