'''
Project 4 - Parallel Page Rank

File: pagerank.py
Description:
  Driver for Project 4

original version: Mandy Chan, Garrett Summers

modified by: Eric Dazet, Nathik Salam
'''

import csv
import parser
import sys
import time
import subprocess
from subprocess import Popen, PIPE, STDOUT
import urlparse

# dictionary that holds the numbers of iterations, nodes, and edges for each input file
# these numbers were either taken from the previous implementation or calculated
def specs():

    # tuple values: number of iterations, number of nodes, number of edges
    
    dictionary = {}
    dictionary.update({"dolphins.csv": [49, 62, 318]})
    dictionary.update({"karate.csv": [38, 34, 156]})
    dictionary.update({"lesmis.csv": [61, 77, 508]})
    dictionary.update({"NCAA_football.csv": [3, 570, 1537]})
    dictionary.update({"polblogs.csv": [71, 1224, 19090]})
    dictionary.update({"stateborders.csv": [93, 51, 214]})
    dictionary.update({"amazon0505.txt": [85, 410236, 3356824]})
    dictionary.update({"p2p-Gnutella05.txt": [21, 8846, 31839]})
    dictionary.update({"soc-sign-Slashdot081106.txt": [64, 77357, 516575]})
    dictionary.update({"wiki-Vote.txt": [38, 7115, 103689]})
    dictionary.update({"soc-LiveJournal1.txt": [67, 4847571, 68993773]})
    
    return dictionary

# sorts and prints pagerank values, applies mapping if a mapping was used while parsing
def printPageRankValues(output, useNames, names):
    mydict={}
    for item in output.split(','):
        spl = item.split(':')
        mydict[int(spl[1])] = spl[0]
        
    for i, w in enumerate(sorted(mydict, key=mydict.get, reverse = True)):
      if i > 20:
        break
      if useNames == True:
         print names[w], mydict[w]
      else:
         print w, mydict[w]

def main():
  fileSpecs = specs()
  
  is_weighted = False      # Used for '-w' flag

  # Setting variable if '-w' is used
  if len(sys.argv) > 1:
    if '-w' in sys.argv:
      is_weighted = True

  # Menu
  print('CSC 466: Lab 3 - PageRank & Link Analysis')
  parse_menu = '1'
  file_name = sys.argv[1]
  version = sys.argv[2]

  # grabs input parameters
  if(file_name[-3:]=='csv'):
    parse_menu = '1'
  elif(file_name[-3:]=='txt'):
    parse_menu = '2'
  else:
    print "Format not supported"
    
  
  # PARSING - CSV Files
  # Note: The algorithm is the same, just parsing is different.
  if parse_menu == '1' or parse_menu == '2':
    if parse_menu == '1':
       print('Parsing/Creating Graph...')
       start = time.time()    # Tracking time
    
       # Parses a csv file and returns a tuple (list, dictionary, dictionary)
       if is_weighted == False:
        (nodes, out_degrees, in_degrees, names) = parser.parse_csv(file_name)
       else:
        (nodes, out_degrees, in_degrees, names) = parser.parse_weighted_csv(file_name)
     
       end = time.time()
       print('Parse/Graph Set-up Time: ' + str(end - start) + ' seconds')

  # PARSING - SNAP Files
    else:
       print('Parsing/Creating Graph...')
       start = time.time()    # Tracking time
    
      # Parses a SNAP file and returns a tuple (list, dictionary, dictionary)
       (nodes, out_degrees, in_degrees, names) = parser.parse_snap(file_name)
   
       end = time.time()
       print('Parse/Graph Set-up Time: ' + str(end-start) + 'seconds')
    
    # truncates name of input file if necessary to index into specs dictionary
    if file_name.rfind('/') != '-1':
       file_name = file_name[file_name.rfind('/') + 1:len(file_name)]

    #
    (numIterations, numNodes, numEdges) = fileSpecs[file_name]

    # determine if a mapping is needed for the final printing of values
    useNames = parse_menu == '1' or file_name == "wiki-Vote.txt"
    
    # Runs Xeon Phi Version
    if version == 'phi' or version == 'both':
       p = subprocess.Popen(['./pr_phi', str(numNodes), str(numEdges), str(numIterations)], stdout=subprocess.PIPE, stdin=subprocess.PIPE)
       
       print "\nPhi Version\n"
       start = time.time()   
         
       # send all nodes pointing to other nodes to stdin
       for node in nodes:
          if in_degrees.get(node) is not None:
             for i in range (0, len(in_degrees[node])):
                p.stdin.write('%d\n' % int(in_degrees[node][i]))
    
       # send the node number, that node's number of in-degrees, and that node's number of out-degrees to stdin
       for node in nodes:
          i = len(in_degrees[node]) if in_degrees.get(node) is not None else 0
          j = out_degrees[node] if out_degrees.get(node) is not None else 0
          p.stdin.write("%d %d %d\n" % (int(node), int(i), int(j)))
       
       # wait for data from C program
       output = p.communicate()[0]
       output = output[:-1]
       p.stdin.close()

       printPageRankValues(output, useNames, names)

       end = time.time()
       print('Phi Total Runtime: ' + str(end - start) + ' seconds')

    # Runs Cuda Version
    if version == 'cuda' or version == 'both':
       p = subprocess.Popen(['./pr_cuda', str(numNodes), str(numEdges), str(numIterations)], stdout=subprocess.PIPE, stdin=subprocess.PIPE)

       print "\nCuda Version\n"
       start = time.time()

       # send all nodes pointing to other nodes to stdin
       for node in nodes:
          if in_degrees.get(node) is not None:
             for i in range (0, len(in_degrees[node])):
                p.stdin.write('%d\n' % int(in_degrees[node][i]))
     
       # send the node number, that node's number of in-degrees, and that node's number of out-degrees to stdin
       for node in nodes:
           i = len(in_degrees[node]) if in_degrees.get(node) is not None else 0
           j = out_degrees[node] if out_degrees.get(node) is not None else 0
           p.stdin.write("%d %d %d\n" % (int(node), int(i), int(j)))
              
       # wait for data from C program
       output = p.communicate()[0]
       output = output[:-1]
       p.stdin.close()

       printPageRankValues(output, useNames, names)
       end = time.time()
       print('Cuda Total Runtime: ' + str(end - start) + ' seconds')
  
  # Wrong input
  else:
    print('Invalid input - exiting')

if __name__ == '__main__':
  main()
