#!/bin/bash

# Ejecutar el script de Python en segundo plano
python3 save_time.py &
# Guardar el PID del proceso de Python
PID_PYTHON=$!

# Inicializar un contador
counter=1

# Ejecutar un bucle que dure 30 segundos
for i in {1..30}
do
  echo $counter
  ((counter++))
  sleep 1
done

# Matar el script de Python después de 30 segundos
kill $PID_PYTHON

# Esperar a que el script de Python termine
wait $PID_PYTHON
echo "El script de Python ha sido detenido."
