#!/bin/bash

# +++ Limpiamos basura
rm -f results.dat
make clean
# +++ Corremos el programa
make
# +++ Especifico la cantidad de hilos
num_threads=4

# +++ Imprimo mensaje
echo #
echo ++++++++++++++++++ FOR $num_threads THREADS ++++++++++++++++++
echo #
# +++ Ejecuto análisis de performance en x corridas y guardo los datos
perf stat -e cpu-clock,cpu-cycles,instructions,cache-references,cache-misses -r 20 ./tiny_mc >> results.dat
# +++ Muestro los datos en salida estandar por terminal
cat results.dat
