#! /bin/bash

make clean
make

echo 10000
time vectorsum /home/clupo/vectors/10000.a /home/clupo/vectors/10000.b
time vectorsum /home/clupo/vectors/10000.a /home/clupo/vectors/10000.b
time vectorsum /home/clupo/vectors/10000.a /home/clupo/vectors/10000.b

echo 100000
time vectorsum /home/clupo/vectors/100000.a /home/clupo/vectors/100000.b
time vectorsum /home/clupo/vectors/100000.a /home/clupo/vectors/100000.b
time vectorsum /home/clupo/vectors/100000.a /home/clupo/vectors/100000.b

echo 1000000
time vectorsum /home/clupo/vectors/1000000.a /home/clupo/vectors/1000000.b
time vectorsum /home/clupo/vectors/1000000.a /home/clupo/vectors/1000000.b
time vectorsum /home/clupo/vectors/1000000.a /home/clupo/vectors/1000000.b

echo 10000000
time vectorsum /home/clupo/vectors/10000000.a /home/clupo/vectors/10000000.b
time vectorsum /home/clupo/vectors/10000000.a /home/clupo/vectors/10000000.b
time vectorsum /home/clupo/vectors/10000000.a /home/clupo/vectors/10000000.b

#echo 100000000
#time vectorsum /home/clupo/vectors/100000000.a /home/clupo/vectors/100000000.b
#time vectorsum /home/clupo/vectors/100000000.a /home/clupo/vectors/100000000.b
#time vectorsum /home/clupo/vectors/100000000.a /home/clupo/vectors/100000000.b
exit
