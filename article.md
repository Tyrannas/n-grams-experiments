# Générer du texte et de la musique avec les modèles n-grammes
Ouvrez votre téléphone et commencez à tapper un message. "Je vais être en". Il y a de fortes chances que votre système de recommandation affiche le mot "retard". Félicitations vous venez très probablement d'utiliser un modèle n-gramme. 
> Mais en quoi consiste exactement un modèle n-gramme?

## 1. Un peu de théorie (désolé pour les maths)

### 1.1 Représenter le langage
Un modèle, comme son nom l'indique, sert à **modéliser**. En l'occurence ce que l'on cherche ici à modéliser c'est le langage. On peut considérer que chaque phrase a une **probabilité d'apparition** dans l'ensemble des phrases possibles, constitués des mots de la langue française. 

> Mais nous on cherche à générer du texte non? Pas à modéliser la langue? 
> C'est vrai, dans l'idée. Mais comme en physique où comprendre comme une planète se déplace permet de prédire sa position future, modéliser la langue permet de créer des structures "cohérentes" avec cette dernière.

Le problème... c'est qu'il est impossible de connaître toutes les phrases existantes, et surtout impossible de connaître leur fréquence d'apparition. 

Une solution serait alors de modéliser la probabilité d'une phrase en fonction de la probabilité conditionnelle des mots la constituant. C'est à dire prédire l'apparition d'un mot n en fonction des probabilité d'apparition de tous les mots de n-1 à 1.

> Par exemple pour une phrase s = "Je suis fatigué." 
>
> P(s) = P("je") * P("suis" | je) * P("fatigué" | ("suis" |"je"))
> 
> Pour plus d'informations: <a href="https://fr.wikipedia.org/wiki/Formule_des_probabilités_composées">Les probabilités composées</a>

### 2.1 Les n-grammes à la rescousse!
Malheureusement cela ne nous aide pas énormément. En effet, que se passe-t-il si l'on cherche à calculer les probabilités des phrases suivantes:
- Je suis fatigué
- Il se fait tard et je suis fatigué
- La journée a été longue et je suis fatigué

Il faudra connaitre à chaque la probabilité de "fatigué" **sachant** tous les mots précédents. Ce qui impliquerait de d'avoir une base énorme de texte pour pouvoir calculer tous les cas particuliers un par un. Et encore, face à une phrase non présente dans l'ensemble d'apprentissage, cela ne marcherait pas.

**Cependant** (et là tout devient facile).

On peut remarquer que dans les trois phrases, les mots précédant "fatigué", sont toujours "je" et "suis".

Et bien le principe du modèle n-gramme, c'est de considérer que cet "historique" de mots précédant "fatigué" peut être approché par un sous ensemble de taille **n** de cet historique.

Mettons que l'on prenne **n = 2** alors pour calculer la probabilité d'apparition de "fatigué" dans la phrase "Il se fait tard et je suis" sera tout simplement P("fatigué" | "suis").

Ce que l'on va alors chercher à faire c'est connaître les fréquences d'apparition de tous les bigrammes de l'ensemble d'apprentissage.
Reprenons nos phrases de tout à l'heure et ajoutons la phrase:
- "Je suis le major Tom et je vais voyager."

Nous pouvons alors calculer les probabilités des bigrammes: 

    B = {P(suis | je) = 4/5}
ecrire les bigrammes
dire comment on predit
dire qu'on va coder
## le code
### structures de données
représentation naive sous forme de dico
dico avec proba
la classe ngramm
attributs
addword
calcproba
generate
### le modèle
attributs, ngramms et n
train
generate
## les exemples
### texte
lotr n = 3
lotr n = 5
3 mousqutaires n = 3
3 mousquetaires n = 4
3 mousquetaires n = 5

dire que c'est que 800 000 mots
### tabs
la recup
le parsing
generer
extraits audio des moins pourries

## pour aller plus loin
dans le code la tokenization, le clean du texte toussa
combiner les markov
les stopwords
le smoothing
la generation avec lstm et openai