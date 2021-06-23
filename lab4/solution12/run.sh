#!/bin/bash

# +++ Limpiamos basura
rm -f results.dat
make clean
# +++ Corremos el programa
make

# +++ Ejecuto 20 corridas y guardo los datos
for i in {1..20};
do
	./tiny_mc >> results.dat
done

# echo #
# echo ++++++++++++++++++       RESULTADOS         ++++++++++++++++++
# echo #
# +++ Muestro los resultados por salida estandar
# cat results.dat
