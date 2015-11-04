'''
Lab 3 - Page Rank

File: parser.py
Description:
  Parser module to parse csv and SNAP files.

Mandy Chan
Garrett Summers
'''

import csv
import sys

nodes = set()           # List of all nodes in the graph
in_degrees = {}         # Dictionary of in-degrees. Key: Node ID, Value: List of
                        #   nodes pointing to it (strings)
out_degrees = {}        # Dictionary of out-degrees. Key: Node ID, Value: Int
                        #   representing the number of nodes it points to.

def parse_snap(file):
  '''
  Parses a snap file and creates a 'graph'. Each snap row is assumed to have the
  format: Node1 [spacing] Node2.
  
  Node1 is always pointing to Node2. Therefore, Node2 gets an in-degree added to
  it while Node1 increments the number of out-degrees it has.
  
  Returns a tuple (list, dictionary [of ints], dictionary [of list of strings])
  '''
  global nodes, in_degrees, out_degrees

  with open(file, 'r') as f:
    for i, line in enumerate(f):
      if line[0] == '#':
        continue
      if i % 1000 == 0:
        print i
      words = line.split()
      nodes.add(words[0])
      nodes.add(words[1])
      if in_degrees.get(words[1]) is None:
        in_degrees[words[1]] = []

      out_degrees[words[0]] = out_degrees.get(words[0], 0) + 1
      in_degrees[words[1]].append(words[0])

  return (nodes, out_degrees, in_degrees)

def parse_weighted_csv(file):
  '''
  See 'parse_csv' function.
  
  The only difference is deciding who gets the in-degree (and out-degree). If 
  N1-val > N2-val, the in-degree is to N1 and the out-degree is to N2 (and 
  vice versa for N2-val > N1-val)
  '''
  global nodes, in_degrees, out_degrees

  with open(file, 'r') as f:
    reader = csv.DictReader(f, fieldnames=['N1', 'N1-val', 'N2', 'N2-val'])
    for row in reader:
      n_one = row['N1'].strip('"')
      n_two = row['N2'].strip('"')

      nodes.add(n_one)
      nodes.add(n_two)

      val_one = int(row['N1-val'].strip())
      val_two = int(row['N2-val'].strip())
      if val_one < val_two:
        out_degrees[n_one] = out_degrees.get(n_one, 0) + 1
        
        if in_degrees.get(n_two) is None:
          in_degrees[n_two] = []
        (in_degrees.get(n_two, [])).append(n_one)
      else:
        out_degrees[n_two] = out_degrees.get(n_two, 0) + 1
        
        if in_degrees.get(n_one) is None:
          in_degrees[n_one] = []
        (in_degrees.get(n_one, [])).append(n_two)
      
  return (nodes, out_degrees, in_degrees)

def parse_csv(file):
  '''
  Parses a csv file and creates a 'graph'. Each csv row is assumed to have the
  format: Node1, Node1-value, Node2, Node2-value. For the most part, we ignore
  the values. 
  
  Node1 is always pointing to Node2. Therefore, Node2 gets an in-degree added to
  it while Node1 increments the number of out-degrees it has.
  
  Returns a tuple (list, dictionary [of ints], dictionary [of list of strings])
  '''

  global nodes, in_degrees, out_degrees

  with open(file, 'r') as f:
    reader = csv.DictReader(f, fieldnames=['N1', 'N1-val', 'N2', 'N2-val'])
    for row in reader:
      nodes.add(row['N1'])
      nodes.add(row['N2'])
      
      if in_degrees.get(row['N2']) is None:
        in_degrees[row['N2']] = []

      # Increment number of out-degrees N1 has
      out_degrees[row['N1']] = out_degrees.get(row['N1'], 0) + 1
      
      # Add 'N1' to N2's list of in-degrees
      (in_degrees.get(row['N2'], [])).append(row['N1'])

  return (nodes, out_degrees, in_degrees)

def main():
  # Testing Purposes
  #parse_snap(sys.argv[1])
  #parse_csv(sys.argv[1])

if __name__ == '__main__':
  main()
