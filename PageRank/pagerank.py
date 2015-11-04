'''
Lab 3 - Page Rank

File: pagerank.py
Description:
  Page rank module that calculates page rank.
  
Mandy Chan
Garrett Summers
'''

import math
import sys                    #Gotta LOVE our globals :D

d = 0.85                      # The chance of going to a random page
episilon = 0.0000000001       # The change in page rank
len_nodes = 0                 # How many nodes there are
out_degrees = {}              # Dictionary of out-degrees. Key: Node, Value: Int
                              #   representing the number of nodes it points to
in_degrees = {}               # Dictionary of in-degrees. Key: Node, Value: List
                              #   of nodes pointing to it (strings)
page_ranks = {}               # Dictionary of current page ranks
previous_ranks = {}           # Dictionary of previous page ranks

def set_up(nodes, out_deg, in_deg):
  '''
  Sets up the page rank structures we will be using, mostly just initializing things
  '''
  global len_nodes, out_degrees, in_degrees, page_ranks

  len_nodes = len(nodes)
  out_degrees = out_deg
  in_degrees = in_deg
  page_ranks = dict.fromkeys(nodes, 1.0/len_nodes)
  previous_ranks = page_ranks.copy()

def vec_length(d):
  '''
  Finds the vector length for page rank
  '''
  sum = 0.0
  for val in d.values():
    sum += val**2

  return math.sqrt(sum)

def page_rank(num_iters):
  '''
  Page Rank algorithm. We keep track of the number of iterations for statistics.
  Initially, all nodes have page rank 1/(number of nodes). Then we use the 
  formula to calculate the new page ranks for all nodes. Afterwards, we compare 
  the vector lengths of the current and previous page ranks. If it changed more 
  than epsilon, repeat the page rank algorithm. Otherwise, print out the top 20 
  nodes and return the number of iterations it took to finish page rank.
  '''
  global page_ranks, previous_ranks
  num_iters += 1

  previous_ranks = page_ranks.copy()

  for key in page_ranks.keys():
    follow_prob = 0

    for i in in_degrees.get(key, []):
      follow_prob += 1.0/out_degrees.get(i)*previous_ranks.get(i)
    page_ranks[key] = ((1 - d)*(1.0/len_nodes) + d*follow_prob)
  
  if math.fabs(vec_length(page_ranks) - vec_length(previous_ranks)) < episilon:
    for i, w in enumerate(sorted(page_ranks, key=page_ranks.get, reverse = True)):
      if i > 20:
        return num_iters
      print w, page_ranks[w]
  else:
    previous_ranks = page_ranks.copy()
    num_iters = page_rank(num_iters)
  return num_iters

def main():
  pass

if __name__ == '__main__':
  main()
