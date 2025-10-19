from tokenizer import BytePairEncoder

BPE = BytePairEncoder()
text = "Ceci est un exemple de texte à encoder avec le Byte Pair Encoding. Le BPE est une technique de tokenization efficace."
train = BPE.train(text, 310)

test = "Ceci est un test de tokenization."

test_ids = BPE.encode(test)
for i in range(len(test_ids)):
    print(f"Token ID: {test_ids[i]}, Byte Sequence: {BPE.decode([test_ids[i]])}")
    
print("Test Encoded IDs:", test_ids)
print("Test Decoded Text:", BPE.decode(test_ids))
