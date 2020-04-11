from collections import defaultdict

def merge_dicts_with_only_lists_as_values(dicts):
  dd = defaultdict(list)
  for d in dicts:
    for key, value in d.items():
      # dict values are always lists for history dicts so extending is fine
      dd[key].extend(value)
  return dd