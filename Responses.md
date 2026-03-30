# Évaluation technique - Scientifique des données
#### Sara Hall
##### 31 Mars, 2026

Code used to generate thee responses is in a notebook located [here](notebooks/test-technique-notebook.ipynb). If this was a full project rather than a proof of concept for a test, I would convert the main bits of code into a py file and only use the notebook for demo purposes, but in the interest of time I'm leaving it as is. I would also put more effort into making the visuals look nice if I were presenting to a client.

# Partie 1 - Préparation des données

## Q1.1 — 3 principaux enjeux de données

**1. Les retours sont encodés comme des valeurs négatives**

Les retours sont encodés comme des quantités négatives (ainsi que des ventes et des profits négatifs), le discount étant identique pour la vente et le retour. ![](assets/distributions.png) J'ai donc regroupé par `customer_id`, `product_id` et `order_id`, puis sommé les quantités, les ventes et les profits, pris la moyenne du discount et retenu la date minimale. Ensuite, j'ai supprimé les lignes où `quantity = 0`, car ce sont celles où la vente et le retour se sont complètement annulés. De cette façon, les retours sont correctement pris en compte sans biaiser les données.

**2. Séries temporelles trop creuses par région × sous-catégorie**

Le jeu de données contient seulement 4 régions et 4 sous-catégories pour environ 2 000 transactions. Une prévision par région et par sous-catégorie produirait des groupes avec aussi peu que 19 mois de données, ce qui n'est pas suffisant pour construire des séries temporelles robustes. ![](assets/heatmap.png) Pour cette raison, j'ai décidé d'agréger les ventes au niveau national par sous-catégorie. De cette façon, chaque groupe dispose de presque 48 mois de données, ce qui permet d'obtenir des séries suffisamment denses pour la modélisation.

**3. Fuite de données dans les colonnes agrégées**

Les colonnes `avg_region_sales` et `avg_city_sales` ne sont rien d'autre que les moyennes globales des ventes par région et par ville sur l'ensemble du jeu de données. Si on les inclut dans les données d'entraînement, le modèle aurait accès à de l'information provenant des données de test, ce qui constitue une fuite de données (*data leakage*). Pour cette raison, ces colonnes ont été supprimées du DataFrame.

## Q1.2 — 2 features les plus influentes

**1. La saisonnalité — lag de 12 mois**

Les ventes présentent une saisonnalité annuelle marquée, notamment pour les chaises, les tables et les bibliothèques. ![Trendlines](assets/trendlines.png) ![Monthly Sales Autocorrelation](assets/acf.png) Pour exploiter ce signal, j'ai calculé des features de lag — en particulier un décalage de 1 et 12 mois — qui permettent au modèle d'utiliser les ventes du même mois de l'année précédente comme prédicteur. Ces lags ont été calculés par sous-catégorie afin d'éviter toute fuite de données entre les groupes.

**2. La sous-catégorie — encodage one-hot**

La sous-catégorie a un impact direct sur le niveau des ventes, chaque type de meuble ayant ses propres dynamiques de demande. C'est également une variable utile pour les gestionnaires de magasins, qui peuvent s'en servir pour anticiper les commandes par type de produit. Pour que le modèle puisse l'utiliser, la variable catégorielle a été transformée via un encodage one-hot (*one-hot encoding*).

## Q1.3 — Actions pour améliorer la qualité des données

**1. Associer les données aux emplacements des magasins**

Les données actuelles semblent refléter les adresses de livraison plutôt que les emplacements des magasins. Chaque magasin dessert une démographie distincte qui pourrait améliorer la précision des prévisions, notamment au niveau du magasin ou du marché local. Il serait utile d'associer chaque transaction à un identifiant de magasin pour capturer ces dynamiques.

**2. Collecter davantage de données historiques**

Avec seulement 4 ans de données, le signal de saisonnalité est limité. Des séries historiques plus longues permettraient au modèle de mieux distinguer les tendances structurelles des variations ponctuelles, et d'améliorer la fiabilité des prévisions.

**3. Introduire un niveau de granularité intermédiaire pour les produits**

Les données contiennent deux niveaux de granularité : le produit individuel et la sous-catégorie. Pour les besoins de la gestion des stocks, il serait utile d'introduire un niveau intermédiaire — par exemple, distinguer les bibliothèques à 5 tablettes de celles à 10 tablettes. Cela permettrait des prévisions plus précises sans aller jusqu'au niveau du SKU individuel.

**4. Enregistrer l'historique des promotions**

Connaître les dates et l'ampleur des promotions passées permettrait de modéliser leur impact sur les ventes et d'améliorer la précision des prévisions lors de futures campagnes.

**5. Assurer l'unicité des `product_id`**

Actuellement, 8 produits distincts partagent le même `product_id`, ce qui complique le suivi des stocks et introduit une ambiguïté dans les données. S'assurer que chaque produit correspond à un identifiant unique améliorerait la fiabilité du catalogue produit et la traçabilité des ventes.

# Partie 2 : Présentation des résultats au client

## Q2.1 - Les résultats de ma solution ML

| | Baseline naïve (lag 12) | Random Forest | Amélioration |
|---|---|---|---|
| MAE | $1,942 | $1,811 | **-$131/mois** |
| RMSE | $2,753 | $2,479 | **-$274** |
| MAPE | 67.5% | 68.1% | — |

**Interprétation pour le client**
Le modèle Random Forest prédit les ventes mensuelles par sous-catégorie avec une erreur moyenne de 1811$ par mois. Pour mettre ce chiffre en contexte, les ventes mensuelles moyennes varient entre 1852$ (Furnishings) et 6660$ (Chairs) — une erreur de 1811$ représente donc environ 27% des ventes moyennes de Chairs, ce qui est raisonnable pour un premier modèle sur 4 ans de données.
Plus important, le modèle surpasse la baseline naïve — c'est-à-dire simplement utiliser les ventes de l'année précédente — sur les métriques MAE et RMSE. Cela démontre que le modèle capte des signaux réels au-delà de la simple saisonnalité.
Valeur ajoutée pour l'entreprise. ![](assets/predictions.png)

En plus, le modéle nous permets de voir les features qui sont plus importants pour preduire les ventes futures. Par example, on peut voir que le chose le plus important sont les vents de l'anée avant. On peut voir aussi que le monthly discount has an effect as well, which is something that store managers can control. ![](assets/feature_importance.png)

Ce modèle est un point de départ. Avec davantage de données historiques et l'ajout des informations promotionnelles recommandées, les performances s'amélioreraient significativement. La valeur immédiate est de démontrer la faisabilité de l'approche et d'établir une baseline solide pour les itérations futures.

## Q2.2 - Comments les gestionnaires vont l'utiliser:

**Concrètement, ce modèle permet aux gestionnaires de magasins de :**

1. Anticiper la demande par sous-catégorie un mois à l'avance, permettant une meilleure gestion des stocks
2. Réduire les ruptures de stock sur les sous-catégories à forte valeur comme Chairs et Tables
3. Quantifier l'impact des rabais sur les ventes futures — le modèle confirme que le rabais moyen est un prédicteur significatif




