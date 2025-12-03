# Usamos una imagen oficial de Jupyter con Spark preinstalado
FROM jupyter/pyspark-notebook:latest

# Cambiamos al usuario root para instalar dependencias del sistema si fueran necesarias
USER root

# Instalamos librerías de Python adicionales definidas en requirements.txt
COPY requirements.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Configuramos el directorio de trabajo
WORKDIR /home/jovyan/work

# Cambiamos de nuevo al usuario sin privilegios para seguridad
USER ${NB_UID}


