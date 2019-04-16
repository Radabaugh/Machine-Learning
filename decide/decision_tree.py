import math
import pickle
import collections

def entropy(data):
  frequency = collections.Counter([item[-1] for item in data])
  def item_entropy(category):
    ratio = float(category) / len(data)
    return -1 * ratio * math.log(ratio, 2)
  return sum(item_entropy(c) for c in frequency.values())


with open("data", "rb") as f:
  L = pickle.load(f)
