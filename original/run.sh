#!/bin/bash

# +++ Limpiamos basura
rm -f results.dat
make clean
# +++ Corremos el programa
make
# +++ Especifico la cantidad de hilos

# +++ Imprimo mensaje
echo #
echo ++++++++++++++++++ SINGLE CORE ++++++++++++++++++
echo #
# +++ Ejecuto análisis de performance en x corridas y guardo los datos
perf stat -e cpu-clock,cpu-cycles,instructions,cache-references,cache-misses -r 20 ./tiny_mc >> results.dat
echo #
echo ++++++++++++++++++       RESULTADOS         ++++++++++++++++++
echo #
# +++ Muestro los resultados por salida estandar
cat results.dat