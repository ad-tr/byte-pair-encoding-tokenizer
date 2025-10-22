import regex as re
import pickle
from pathlib import Path

class BytePairEncoder:
  def __init__(self):
    self.merges = {}
    self.vocab = {}
    
  def train(self, text, vocab_size):
    if vocab_size < 257:
      print("Veuilez indiquer une taille de vocabulaire supérieure à 256 (chars unicode déjà présents)")
      return
    num_merges = vocab_size - 256
    merges = {}
    current_ids = self._text_to_bytes(text)
    for i in range(num_merges):
      stats = self._get_stats(current_ids)
      top_pair = max(stats, key=stats.get)

      merges[top_pair] = 256 + i
      current_ids = self._merge(current_ids, top_pair, 256 + i)
    self.merges = merges
    self._create_vocab_with_merges()

  def encode(self, text):
    tokens = self._text_to_bytes(text)
    merges = self.merges
    while len(tokens) >= 2:
        stats = self._get_stats(tokens)
        pair = min(stats, key=lambda p: merges.get(p, float("inf")))
        if pair not in merges:
            break
        idx = merges[pair]
        tokens = self._merge(tokens, pair, idx)
    return [token for sublists in tokens for token in sublists]

  def decode(self, ids):
    tokens = b"".join(self.vocab[idx] for idx in ids)
    return tokens.decode("utf-8")
  
  def save(self, file_name):
    path = Path(__file__).resolve().parents[1] / "data" / file_name
    with open(path, "wb") as f:
      pickle.dump(self.merges, f)
    print(f"Merges sauvegardé avec succès dans {path}")
      
  def load(self, file_name):
    path = Path(__file__).resolve().parents[1] / "data" / file_name
    with open(path, "rb") as f:
      self.merges = pickle.load(f)
    self._create_vocab_with_merges()
    print(f"Merges chargé avec succès depuis {path}")
    print("Vocab mis a jour avec succès")
    
  def _merge(self, ids, pair ,idx):
    new_ids = []
    for word in ids:
      new_sub_ids = []
      i = 0
      while i < len(word):
        if i < len(word) - 1 and word[i] == pair[0] and word[i+1] == pair[1]:
          new_sub_ids.append(idx)
          i += 2
        else:
          new_sub_ids.append(word[i])
          i += 1
      new_ids.append(new_sub_ids)
    return new_ids
    
  def _text_to_bytes(self, text):
    gpt2pat = re.compile(r"""\[speaker\d{3}:\]| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
    splited_text = re.findall(gpt2pat, text)
    ids = [list(map(int, item.encode('utf-8'))) for item in splited_text]
    return ids
        
  def _get_stats(self, ids):
    counts = {}
    for item in ids:
        for pair in zip(item, item[1:]):
          counts[pair] = counts.get(pair, 0) + 1
    return counts
  
  def _create_vocab_with_merges(self):
    vocab = {idx: bytes([idx]) for idx in range(256)}
    for (p0, p1), idx in self.merges.items():
        vocab[idx] = vocab[p0] + vocab[p1]
    self.vocab = vocab