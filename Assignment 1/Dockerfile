# FROM <image> - this specifies the base image that the build will extend.
# This sets the base environment.
FROM python:3.12-slim 
# Uses a slim version of Python 3.12 - lightweight but good enough to
# install everything else you need (like Spark, Java, Jupyter, etc.)

# ENV <name> <value> - this instruction sets an environment variable 
# that a running container will use. 
ENV DEBIAN_FRONTEND=noninteractive 
# Prevents interactive prompts during package installs. Makes the 
# Docker build automated and smooth. 

# RUN <command> - this instruction tells the builder to run the specified
# comnmand. 
RUN apt-get update && \
    apt-get install -y --no-install-recommends openjdk-17-jdk-headless procps bash && \
    rm -rf /var/lib/apt/lists/* && \
    ln -sf /bin/bash /bin/sh
# Installs OpenJDK 17 -> required to run PySpark. 
# Forces /bin/sh to use bash -> Spark scripts expect bash, not dash (the debian default)

ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PATH=$PATH:$JAVA_HOME/bin
# Tells Spark where to find Java.
# Updates system PATH so java works from anywhere inside the container. 

# WORKDIR <path> - this instruction specifies the 'working directory' or
# the path in the image where files will be copied and commands will be 
# executed 
WORKDIR /app
# Sets the current working directory in the container to /app.
# When you COPY, install, or run commands, they happen relative to /app 

# COPY <host-path> <image-path> - this instruction tells the builder 
# to copy files from the host and put them into the container image. 
COPY requirements.txt ./
# Copies your requirements.txt file into the container. 
# So it can install your Python libraries (like pyspark, pandas, etc.)

RUN pip install --no-cache-dir -r requirements.txt 
# Installs all Python dependencies. These are needed for your project 
# (PySpark, Delta Lake, Pandas, etc.)

# EXPOSE <port-number> - this instruction sets configuartion on the image
# that indicates a port the image would like to expose. 
EXPOSE 8888
# Lets you access JupyterLab from your browser (e.g. localhost:8888)
# Jupyter runs inside the container and you need to expose the port. 

# Create a volume mount point for notebooks
VOLUME /app 
# Marks /app as a volume. This helps with mounting your code directory 
# from your machine into the container, so any changes you make are visible
# in real time. 

# Enable JupyterLab via environment variable
ENV JUPYTER_ENABLE_LAB=yes
# Switches from classic Jupyter to JupyterLab, a more modern interface.
# It's easier to manage multiple notebooks and files in JupyterLab. 

# CMD ['<command>', '<arg1>'] - this instruction sets the default command
# a container using this image will run 
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--notebook-dir=/app"]
# This command runs JupyterLab inside the container. Makes it accessible
# from your local browser at http://localhost:8888. 