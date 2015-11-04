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

def main():
  '''
  The menu shown at the beginning of the Page Rank program. It asks whether the
  file is a csv or snap file, then the file name.
   
  Ex:
    > python lab3.py [-w]
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
  
  # PARSING - CSV Files
  # Note: The algorithm is the same, just parsing is different.
  if parse_menu == '1':
    print('Parsing/Creating Graph...')
    start = time.time()    # Tracking time
    
    # Parses a csv file and returns a tuple (list, dictionary, dictionary)
    if is_weighted == False:
      (nodes, out_degrees, in_degrees) = parser.parse_csv(file_name)
    else:
      (nodes, out_degrees, in_degrees) = parser.parse_weighted_csv(file_name)
      
    end = time.time()
    print('Parse/Graph Set-up Time: ' + str(end - start) + ' seconds')

    # Sets up page rank structures
    pagerank.set_up(nodes, out_degrees, in_degrees)

    # PAGE RANKING
    print('Page Ranking...')
    start = time.time()
    num_iters = pagerank.page_rank(0)  # Stores # of page rank iterations
    end = time.time()
    
    # Statistics
    print('Page Rank Time: ' + str(end-start) + ' seconds')
    print('Page Rank Iterations: ' + str(num_iters))

  # PARSING - SNAP Files
  elif parse_menu == '2':
    print('Parsing/Creating Graph...')
    start = time.time()    # Tracking time
    
    # Parses a SNAP file and returns a tuple (list, dictionary, dictionary)
    (nodes, out_degrees, in_degrees) = parser.parse_snap(file_name)
    
    end = time.time()
    print('Parse/Graph Set-up Time: ' + str(end-start) + 'seconds')

    # Sets up page rank structures
    pagerank.set_up(nodes, out_degrees, in_degrees)

    # PAGE RANKING
    print('Page Ranking...')
    start = time.time()
    num_iters = pagerank.page_rank(0)  # Stores # of page rank iterations
    end = time.time()
    
    # Statistics
    print('Page Rank Time: ' + str(end-start) + ' seconds')
    print('Page Rank Iterations: ' + str(num_iters))
  
  # Wrong input
  else:
    print('Invalid input - exiting')

if __name__ == '__main__':
  main()
