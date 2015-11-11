'''
Lab 3 - Page Rank

File: lab3.py
Description:
  Condenses all operations for Lab 3 here. Similar to a control center.

Mandy Chan
Garrett Summers
'''

import csv
import pagerank
import parser
import sys
import time
import subprocess
from subprocess import Popen, PIPE, STDOUT
import urlparse
from collections import OrderedDict
from operator import itemgetter

def specs():
    
    dictionary = {}
    dictionary.update({"dolphins.csv": [49, 62, 318]})
    dictionary.update({"karate.csv": [38, 34, 156]})
    dictionary.update({"lesmis.csv": [61, 77, 508]})
    dictionary.update({"NCAA_football.csv": [3, 570, 1537]})
    dictionary.update({"polblogs.csv": [71, 1224, 19090]})
    dictionary.update({"stateborders.csv": [93, 51, 214]})
    dictionary.update({"amazon0505.txt": [85, 410236, 3356824]})
    dictionary.update({"p2p-Gnutella05.txt": [21, 8846, 31839]})
    dictionary.update({"soc-sign-Slashdot081106.txt": [64, 77350, 516575]})
    dictionary.update({"wiki-Vote.txt": [38, 7115, 103689]})
    dictionary.update({"soc-LiveJournal1.txt": [67, 4847571, 68993773]})
    
    return dictionary

def printPageRankValues(output, useNames, names):
    mydict={}
    for item in output.split(','):
        spl = item.split(':')
        if useNames == True:
            mydict[names[int(spl[1])]] = spl[0]
        else:
            mydict[int(spl[1])] = spl[0]

    mydict = OrderedDict(sorted(mydict.items(), key=itemgetter(1), reverse=True))

    i = 0;
    for x,w in mydict.iteritems():
        if i > 20:
            break
        print str(x), str(w)
        i += 1

def main():
  '''
  The menu shown at the beginning of the Page Rank program. It asks whether the
  file is a csv or snap file, then the file name.
   
  Ex:
    > python lab3.py [-w, -c]
    CSC 466: Lab 3 - Page Rank & Link Analysis
    Parse:
      1) csv
      2) snap
    (User enters 1 or 2)
    File name: (User enters file name here)
   
  There is an optional flag '-w' that is used for the Football csv. The program
  outputs every 1000 lines (to ensure that it's parsing) and then at the end of
  the page rank algorithm, print out the top 20 nodes and how long it took to 
  calculate page rank.
   
  Note: -w doesn't quite work at the moment. Please ignore it for now.
  '''
    
  fileSpecs = specs()
  
  is_weighted = False      # Used for '-w' flag

  # Setting variable if '-w' is used
  if len(sys.argv) > 1:
    if sys.argv[1] == '-w':
      is_weighted = True

  # Menu
  print('CSC 466: Lab 3 - PageRank & Link Analysis')
  parse_menu = raw_input('Parse:\n' +
                         '1) csv\n' +
                         '2) snap\n'
                        )
  file_name = raw_input('File name: ')
  version = raw_input('Xeon Phi, Cuda or Both? (x/c/b): ')
  
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
    
    '''
    for node in nodes:
      for i in range (0, len(in_degrees[node])):
        print str(node) + " " + str(in_degrees[node][i])
    
    for node in nodes:
      print str(out_degrees[node])
    '''
    if file_name.rfind('/') != '-1':
       file_name = file_name[file_name.rfind('/') + 1:len(file_name)]

    (numIterations, numNodes, numEdges) = fileSpecs[file_name]
    print numIterations, numNodes, numEdges
    
    '''
    Call C Program
    '''
    
    if version == 'x' or version == 'b':
       p = subprocess.Popen(['./pr_phi', str(numNodes), str(numEdges), str(numIterations)], stdout=subprocess.PIPE, stdin=subprocess.PIPE)
       
       for node in nodes:
          print "Node: " + str(node)
          if out_degrees.get(node) is not None: print str(out_degrees[node])
          if out_degrees.get(node) is None: print "None"
          if in_degrees.get(node) is not None:
             for i in range (0, len(in_degrees[node])):
                p.stdin.write('%d %d\n' % (int(node), int(in_degrees[node][i])))
             
       output = p.communicate()[0]
       output = output[:-1]
       p.stdin.close()
       
       useNames = parse_menu == '1' or file_name == "wiki-Vote.txt"
       print str(useNames)
       printPageRankValues(output, useNames, names)

    if version == 'c' or version == 'b':
       p = subprocess.Popen(cuda_command, stdout=subprocess.PIPE, stdin=subprocess.PIPE)
       '''
       for node in nodes:
          for i in range (0, len(in_degress[node])):
             p.stdin.write('%d %d\n' % (int(node), int(in_degrees[node][i])))
       '''
    
    # Sets up page rank structures
    pagerank.set_up(nodes, out_degrees, in_degrees)

    # PAGE RANKING
    print('Page Ranking...')
    start = time.time()
    num_iters = pagerank.page_rank(0, names, parse_menu)  # Stores # of page rank iterations
    end = time.time()
    
    # Statistics
    print('Page Rank Time: ' + str(end-start) + ' seconds')
    print('Page Rank Iterations: ' + str(num_iters))
    
  
  # Wrong input
  else:
    print('Invalid input - exiting')

if __name__ == '__main__':
  main()
