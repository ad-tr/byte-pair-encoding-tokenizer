from tokenizer import BytePairEncoder

# Utilisation générale
BPE = BytePairEncoder()
with open("./data/wikipedia-fr-sample.txt", "r") as f:
    text = f.read()

## Nous definissons le nombre de tokens que nous voulons dans notre vocabulaire.
BPE.train(text, 400)
## Ensuite, nous pouvons sauvegarder les merges qui ont été effectués dans un fichier pickle.
BPE.save("merges.pkl")
## Pour la prochaine fois je pourrais utiliser BPE.load("merges.pkl") sans devoir utiliser BPE.train()

## Test d'encode et de decode
test = "Ceci est un mistral venant du nord et traversant le sud, il est chaud et lourd."
test_ids = BPE.encode(test)
print("Test Encoded IDs:", test_ids)
print("Test Decoded Text:", BPE.decode(test_ids))
print(f"Le ratio de compression est de: {len(test_ids)/len(test)}")



