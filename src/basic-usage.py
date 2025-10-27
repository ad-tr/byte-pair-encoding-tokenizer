from tokenizer import BytePairEncoder
import json

# Utilisation générale
BPE = BytePairEncoder()
with open("./data/wikipedia-fr-sample.txt", "r") as f:
    text = f.read()

## Nous definissons le nombre de tokens que nous voulons dans notre vocabulaire.
BPE.train(text, 400)
## Ensuite, nous pouvons sauvegarder les merges qui ont été effectués dans un fichier pickle.
BPE.save("merges.pkl")
## Pour la prochaine fois je pourrais utiliser BPE.load("merges.pkl") sans devoir utiliser BPE.train()

## Variables à encoder
test = "Ceci est un mistral venant du nord et traversant le sud, il est chaud et lourd."

json_file = open('./data/conversation/conversation.json')
json_str = json_file.read()
json_data = json.loads(json_str)

## Test d'encode et de decode
test_json = BPE.encode(json_data["conversations"][0], "conversation")
test_ids = BPE.encode(test, "document")

print("------------------------------------")
print("Test Encoded IDs:", test_ids)
print("Test Decoded Text:", BPE.decode(test_ids))
print("------------------------------------")
print("Test Encoded JSON IDs:", test_json)
print("Test Decoded JSON IDs:", BPE.decode(test_json))
print("------------------------------------")
print(f"Le ratio de compression est de: {len(test_ids)/len(test)}")