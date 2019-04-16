import math
import pickle
import operator
import collections

def entropy(data):
  frequency = collections.Counter([item[-1] for item in data])
  def item_entropy(category):
    ratio = float(category) / len(data)
    
    return -1 * ratio * math.log(ratio, 2)

  return sum(item_entropy(c) for c in frequency.values())


def best_feature_for_split(data):
  baseline = entropy(data)
  def feature_entropy(feature):
    def e(v):
      partitioned_data = [d for d in data if d[feature] == v]
      proportion = (float(len(partitioned_data)) / float(len(data)))

      return proportion * entropy(partitioned_data)

    return sum(e(v) for v in set([d[feature] for d in data]))
  features = len(data[0]) - 1
  information_gain = [baseline - feature_entropy(feature) for feature in range(features)]
  best_feature, best_gain = max(enumerate(information_gain), key=operator.itemgetter(1))

  return best_feature


def potential_leaf_node(data):
  count = collections.Counter([i[-1] for i in data])

  return count.most_common(1)[0] # The top item


def create_tree(data, label):
  category, count = potential_leaf_node(data)

  if count == len(data):
    return category

  node = {}
  feature = best_feature_for_split(data)
  feature_label = label[feature]
  node[feature_label] = {}
  classes = set(d[feature] for d in data)

  for klass in classes:
    partitioned_data = [d for d in data if d[feature] == klass]
    node[feature_label][klass] = create_tree(partitioned_data, label)
  
  return node


def classify(tree, label, data):
  root = list(tree.keys())[0]
  node = tree[root]
  index = label.index(root)
  for key in node.keys():
    if data[index] == key:
      if isinstance(node[key], dict):
        return classify(node[key], label, data)
      else:
        return node[key]


def as_rule_str(tree, label, ident=0):
  space_ident = '  '*ident
  s = space_ident
  root = list(tree.keys())[0]
  node = tree[root]
  index = label.index(root)
  for key in node.keys():
    s += 'if ' + label[index] + ' = ' + str(key)
    if isinstance(node[key], dict):
      s += ':\n' + space_ident + as_rule_str(node[key], label, ident + 1)
    else:
      s += ' then ' + str(node[key]) + ('.\n' if ident == 0 else ', ')
  if s[-2:] == ', ':
    s = s[:-2]
  s += '\n'

  return s


def find_edges(tree, label, X, Y):
  X.sort()
  Y.sort()
  diagonals = [i for i in set(X).intersection(set(Y))]
  diagonals.sort()
  L = [classify(tree, label, [diagonal, diagonal]) for diagonal in diagonals]
  low = L.index(False)
  min_x = X[low]
  min_y = Y[low]

  high = L[::-1].index(False)
  max_x = X[len(X)-1 - high]
  max_y = Y[len(Y)-1 - high]

  return (min_x, min_y), (max_x, max_y)


with open("data_rand", "rb") as f:
  L = pickle.load(f)

data = [[0, 0, False], [-1, 0, True], [1, 0, True], [0, -1, True], [0, 1, True]]
label = ['x', 'y', 'out']

tree = create_tree(L, label)
print(as_rule_str(tree, label))
print(classify(tree, label, [1, 1]))
print(classify(tree, label, [1, 2]))
