import subprocess
from subprocess import Popen, PIPE, STDOUT

p = subprocess.Popen(['./a.out'],stdout=subprocess.PIPE,stdin=subprocess.PIPE)
p.stdin.write('1 3\n')
p.stdin.write('2 3\n')
print p.communicate()[0]
p.stdin.close()
