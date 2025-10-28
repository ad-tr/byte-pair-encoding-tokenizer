import regex as re
import pickle
from pathlib import Path
import json

class BytePairEncoder:
  """Byte-Pair Encoding algorithm."""
  
  def __init__(self):
    self.merges = {}
    self.vocab = {idx: bytes([idx]) for idx in range(256)}
    self.stats = {}
    self.special_tokens = {
      '<begin_of_text>': 100001,
      '<end_of_text>': 100002,
      '<im_start>': 100003,
      '<im_sep>': 100004,
      '<im_end>': 100005,
    }
    
  def train(self, text, vocab_size):
    """
    Train the BPE tokenizer from scratch.

    Args:
        text (str): The training text.
        vocab_size (int): The desired vocabulary size (give a value up to 256).
    """
    if vocab_size < 257:
      raise ValueError("La taille du vocabulaire doit être supérieure à 256")
    num_merges = vocab_size - 256
    merges = {}
    current_ids = self._text_to_bytes(text)
    for i in range(num_merges):
      stats = self._get_stats(current_ids)
      try: 
        top_pair = max(stats, key=stats.get)
      except Exception as e:
        if "max()" in str(e) and "empty" in str(e):
          raise ValueError("Error: La taille du texte d'entrainement est trop petite pour atteindre le vocabulaire demandé")
      merges[top_pair] = 256 + i
      current_ids = self._merge(current_ids, top_pair, 256 + i)
    
    self.merges = merges
    self._create_vocab_with_merges()

  def encode(self, text, mode):
    """
    Encode the input model into a list of IDs

    Args:
        mode (str: "conversation"|"document" )
        text (str): The text to encode.
        
    Returns:
        int[]: List of token IDs.
    """
    if mode == "document":
      text = f"<begin_of_text>{text}<end_of_text>"
    elif mode == "conversation":
      conversation = []
      for message in text:
        conversation.append(f"<im_start>{message['role']}<im_sep>{message['message']}<im_end>")
      text = ''.join(conversation)

    tokens = self._text_to_bytes(text)
    for pair, idx in sorted(self.merges.items(), key=lambda x: x[1]):
        tokens = self._merge(tokens, pair, idx)
    return [token for sublists in tokens for token in sublists]

  def decode(self, ids):
    """
    Decode the IDs give in input into string

    Args:
        ids (int[]): A list of tokens created with encode()
    
    Returns:
        str: The decoded text.
    """
    for id in ids:
      if id not in self.vocab:
        raise ValueError(f"{id} n'existe pas dans le vocabulaire")
    tokens = b"".join(self.vocab[idx] for idx in ids)
    return tokens.decode("utf-8")
  
  def save(self, path):
    """
    Save the learned merges to a file with .pkl extension in ./data/save/.
    
    Args:
        path (str): The path and the name of the file with .pkl (Example: "data/merges.pkl")
    """
    with open(path, "wb") as f:
      pickle.dump(self.merges, f)
    print(f"Merges sauvegardé avec succès dans {path}")
      
  def load(self, path):
    """
    Load previously saved merges from a .pkl file.

    Args:
        path (str): The path and the name of the file with .pkl (Example: "data/merges.pkl")
    """
    with open(path, "rb") as f:
      self.merges = pickle.load(f)
    self._create_vocab_with_merges()
    print(f"Merges chargé avec succès depuis {path}")
    print("Vocab mis a jour avec succès")
    
  def _merge(self, ids, pair ,idx):
    """
    Merge all occurrences of a token pair into a new token.

    Args:
        ids (int[][]): List of lists of IDs representing tokenized words.
        pair ((int, int)): The pair of tokens to replace.
        idx (int): The ID of the new token that replaces the pair.
    
    Returns:
        int[][]: Updated IDs after merging.
    """
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
      """
      Convert text to bytes with utf-8, add special tokens and use GPT-2 regex for word splitting

      Args:
          text (str): The text to encode
      
      Returns:
          ids[][]: The text encoded with utf-8 with special tokens and splitted.
      """
      placeholder_to_id = {}
      
      for i, (token, token_id) in enumerate(self.special_tokens.items()):
          placeholder = chr(0xE000 + i)
          placeholder_to_id[placeholder] = token_id
          text = text.replace(token, placeholder)
      
      placeholder_chars = ''.join(placeholder_to_id.keys())
      split_pattern = f'([{re.escape(placeholder_chars)}])'
      parts = re.split(split_pattern, text)
      gpt2pat = re.compile(r""" ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
      
      ids = []
      for part in parts:
          if not part:
              continue
          
          if part in placeholder_to_id:
              ids.append([placeholder_to_id[part]])
          else:
              splited_text = re.findall(gpt2pat, part)
              for item in splited_text:
                  ids.append(list(map(int, item.encode('utf-8'))))
      
      return ids
        
  def _get_stats(self, ids):
    """
    Calculate frequency statistics for all consecutive token pairs.

    Args:
        ids (int[][]): List of lists of IDs representing tokens.
    
    Returns:
        dict: Dictionary with pairs as keys and their frequencies as values.
    """
    counts = {}
    for item in ids:
        for pair in zip(item, item[1:]):
          counts[pair] = counts.get(pair, 0) + 1
    return counts
  
  def _create_vocab_with_merges(self):
    """
    Rebuild the complete vocabulary by applying all learned merges.
    
    Updates self.vocab with the new tokens created by the merges.
    """
    vocab = {idx: bytes([idx]) for idx in range(256)}
    for (p0, p1), idx in self.merges.items():
        vocab[idx] = vocab[p0] + vocab[p1]
        
    for st,st_id  in self.special_tokens.items():
      vocab[st_id] = st.encode('utf-8')
      
    self.vocab = vocab