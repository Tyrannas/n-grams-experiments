# Générer du texte et de la musique avec les modèles n-grammes

Ouvrez votre téléphone et commencez à tapper un message. "Je vais être en". Il y a de fortes chances que votre système de recommandation affiche le mot "retard". Félicitations vous venez très probablement d'utiliser un modèle n-gramme. 
> Mais en quoi ça consiste exactement un modèle n-gramme?
---
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

Et bien le principe du modèle n-gramme, c'est de considérer que cet "historique" de mots précédant "fatigué" peut être approché par un sous ensemble de taille **n** de cet historique. On peut alors parler de chaîne de Markov d'ordre n - 1. 

> Une hypothèse de Markov est le fait de considérer que certaines suites d'événements peuvent être expliquées de façon complètement indépendante, ou bien en utilisant un historique réduit. On parle  d'ordre n - 1 car on utilise les n - 1 mots précédant le mot n pour prédire la probabilité de ce dernier. 
Pour plus d'informations: <a href="https://en.wikipedia.org/wiki/Markov_chain">Les chaines de Markov</a> (voir chaines avec mémoire).

Mettons que l'on prenne **n = 2** alors pour calculer la probabilité d'apparition de "fatigué" dans la phrase "Il se fait tard et je suis" sera tout simplement P("fatigué" | "suis").

Ce que l'on va alors chercher à faire c'est connaître les fréquences d'apparition de tous les bigrammes de l'ensemble d'apprentissage.
Reprenons nos phrases de tout à l'heure et ajoutons la phrase:
- "Je m'appelle Tom et je vais dans l'espace."

Nous pouvons alors calculer les probabilités des bigrammes contenant par exemple le mot "je": 

    S[je] = {
        P(suis | je) = 3/5
        P(vais | je) = 1/5
        P(m'appelle | je) = 1/5
    }

Cela revient à dire que si l'on rencontre le mot "je" dans une phrase, il y a 60% de chances qu'il soit suivit du mot "suis", et 20% de chances qu'il soit suivi de "vais" ou "m'appelle". 

Le système de génération de texte est alors enfantin. En prenant la fonction de répartition du bigramme ci dessus, avec x le mot à prédire:

    Fx(x) = {
        0
        P[X <= "suis"] = 3/5
        P[X <= "vais"] = 4/5
        P[X <= "m'appelle"] = 5/5
    }

Il suffit de tirer un nombre aléatoire **y** entre 0 et 1, et de choisir le mot tel que **y** soit compris entre deux bornes de la fonction de répartition.

> Ex: si je tire 0.70, 3/5 < 0.70 < 4/5, le mot généré sera donc "vais".

Bien sûr, prédire un mot en prenant seulement en compte le mot d'avant est assez réducteur et le résultat sera souvent assez limité, mais dès que l'on utilise des valeurs de **n** égales à 4 ou 5, la cohérence du texte généré augmente. 

> **Attention !** Plus **n** augmente, plus le contenu généré risquera de coller au texte originel et de ne plus rien proposer d'original. C'est ce qu'on appelle communément en machine learning **l'overfitting.**
>
> Il faut donc faire attention en choississant **n**, on pourra se permettre des valeurs plus élevées sur des corpus d'entraînement plus gros. Mais généralement les 4-grammes (comme un breton à 9h du matin) ou 5-grammes donnent des résultats corrects.

### 2.3 Cas d'usage

Une brève parenthèse avant de passer au code lui même, pour parler des cas d'usages 'réels' de ces modèles. 
Les modèles n-grammes sont utilisés plus souvent qu'on ne peut le penser. Notamment en complétion des algorithmes de reconnaissance automatique du langage écrit ou parlé, ou même encore pour aider à la reconnaissance de patterns lors du séquençage ADN.

> Imaginez un logiciel de reconnaissance de caractères (OCR) à qui l'on donnerait la phrase suivante en entrée "J'ai mangé des pêches", et que le dernier mot fort mal écrit, soit reconnu à 70% comme "bêches" et à 30% comme pêches. Notre algorithme pourra alors tenter de s'appuyer sur un modèle 4-grammes (toujours aucun rapport avec le chouchen), pour infirmer sa déduction puisque le mot bêche n'apparait jamais après la séquence "ai mangé des".

En réalité, ce type de modèle est loin de s'appliquer simplement aux mots, et peut être utilisé pour toute suite logique de n éléments au sein d'un ensemble, cela peut être des lettres (pour prédire caractère par caractère ce qui va être écrit), avec des sons, des suites de pixels... Le système est simple, es possibilités nombreuses.

-----

Ces bases théoriques étant posées, et le principe de génération par n-gramme clair comme de l'eau de roche. Passons maintenant à une partie plus amusante, la pratique.

## 2. Implémentation en python

Est décrite ci dessus une implémentation assez naïve du modèle, et en pseudo code. Pour ceux voulant aller plus loin ou tester eux même, le code complet avec des méthodes pour preprocesser le texte est disponible <a href="https://github.com/Tyrannas/n-grams-experiments">ici.</a>

Bien.
Résumons donc ce dont nous avons besoin pour faire fonctionner notre modèle:

- Un corpus d'apprentissage pour calculer les probabilités.
- Une méthode d'apprentissage qui pour chaque n-gramme calculera lesdites probabilités.
- Une méthode de génération une fois l'apprentissage terminé.

### 2.1 Les structures de données

La première question qui se pose est celui de la représentation des n-grammes. Pour rappel, il faut pouvoir associer des probabilités pour le nième mot de suivre n - 1 mots.

Une solution est donc pour chacune de ces suites de n - 1 mots (que nous pourrons appeler ici (n-1)-gramme) de tenir un dictionnaire de mots à 'prédire' associant le mot à sa probabilité de suivre la suite. Ainsi pour **n = 4** la suite 'homme sois plus' aura pour dictionnaire:

    {
        'violent': 0.33,
        'puissant': 0.33,
        'ardent': 0.33
    }

Dans les faits, l'apprentissage se fera au fur et à mesure, il faudra donc être en mesure de recalculer les probabilités si l'on ajoute un nouveau mot au dictionnaire. Or comment calculer cette probabilité?

    P(mot) = nb_occurences_mot / nb_total_occurences

Pour cela, on rajoute aussi la notion d'occurence dans le dictionnaire:

    {
        'violent': {'count': 2, 'proba': 0.33},
        'puissant': {'count': 2, 'proba': 0.33},
        'ardent': {'count': 2, 'proba': 0.33}
    }
Et on gardera en parallèle pour chaque suite de mots un compte du total des count du dictionnaire (qui pourrait aussi être recalculé à chaque fois).

Ainsi, si l'on rencontre à nouveau le mot 'violent' après la suite "homme sois plus", on obtiendra:

    {
        'violent': {'count': 3, 'proba': 0.43},
        'puissant': {'count': 2, 'proba': 0.29},
        'ardent': {'count': 2, 'proba': 0.29}
    }

Il suffit de procéder de manière identique pour chaque (n-1)-gramme unique trouvé dans le texte.

### 2.1 L'apprentissage

On utilisera ici deux classes:
- une classe NGram chargée de la mise à jour du dictionnaire abordé plus tôt. Nous créerons donc une instance de NGram par (n-1)-gramme trouvé.
- une classe Model chargée de l'apprentissage global et du management des instances NGram.

Une fois cela établi l'implémentation en est assez simple.

```python
class NGram:
    def __init__(self):
        self.words_to_predict = {}
    
    def add_word(self, word):
        # si le mot existe déjà on incrémente son count
        # sinon on créé le mot avec un count de 1
        # on recalcule les probas
    
    def compute_proba(self):
        # total_count = somme des count de chaque mot de self.words_to_predict
        # pour chaque mot de self.words_to_predict:
            # sa proba vaut son propre count / total_count


class Model:
    def __init__(self, n):
        # n la taille du n-gramme
        # comme on prendra des suites de mots de taille n - 1, pour s'éviter de réécrire n - 1 on assigne self.n = n - 1
        self.n = n - 1
        self.ngrams = {}
    
    def train(self, texte):
        # on sépare le texte en tableau de mots en splittant sur les espaces
        # pour i = self.n jusqu'à la fin du texte:
            # on prend self.n mots à partir de i
            # on id composé de ces mots pour identifier le ngramme
            # on créé self.ngrams[id] = Ngram() s'il n'existe pas
            # on utilise la méthode add_word du ngramme nouvellement créé en lui passant le mot suivant les self.n mots:
            # ==> self.ngrams[id].addword[texte[i + self.n + 1]]
        
```

Et voilà, une fois le texte parcouru en entier par la méthode train, toutes vos probabilités devraient être calculées, et il ne reste plus qu'à générer du texte.

### 2.3 La génération

Reprenons nos deux classes et ajoutons les méthodes de génération:
- la classe NGram a besoin de tirer un nombre aléatoire et de renvoyer un mot.
- la classe Model elle doit sélectionner le bon objet Ngram en fonction de la séquence d'entrée pour générer un mot.

```python
class NGram:
    ...
    def generate(self):
        # on tire un nombre entre 0 et 1
        # on sélectionne le mot dont la probabilité cumulée est supérieure à ce nombre et on renvoie le mot

class Model:
    def generate(self, words):
        # on cree un id à partir des mots
        # on teste si cet id est enregistré dans les ngrammes
        # si oui on demande à ce ngramme de générer un mot
```

Et voilà, c'était extrêmement simple. On peut ensuite modifier un peu la méthode generate de la classe Model pour générer automatiquement de plus longues suites de mots en créant à chaque fois une nouvelle séquence en enlevant le premier mot de la séquence d'avant, et en rajoutant le mot qui vient d'être prédit.

    "Major Tom to" ==> "Ground"
    "Tom to Ground" ==> "Control"
    "to Ground Control" ==> ...
    etc.

Et maintenant testons un peu ce modèles avec des exemples concrets.
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
