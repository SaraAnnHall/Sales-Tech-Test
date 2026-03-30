# Évaluation technique - Scientifique des données
#### Sara Hall
##### 31 Mars, 2026

Code used to generate thee responses is in a notebook located [here](notebooks.test-technique-notebook.ipynb). If this was a full project rather than a proof of concept for a test, I would convert the main bits of code into a py file and only use the notebook for demo purposes, but in the interest of time I'm leaving it as is. 

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

