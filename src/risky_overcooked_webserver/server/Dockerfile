FROM python:3.11-slim

ARG BUILD_ENV
ARG OVERCOOKED_BRANCH
ARG GRAPHICS

WORKDIR /app

RUN apt-get -y update
RUN apt-get upgrade -y
RUN apt-get install -y git
RUN git clone --recursive https://github.com/mosmith3asu/risky_overcooked.git --branch $OVERCOOKED_BRANCH --single-branch /risky_overcooked
ENV PYTHONPATH="${PYTHONPATH}:/risky_overcooked/src"

# Install bigger dependancies
RUN apt-get install -y libgl1-mesa-dev
RUN pip install opencv-python-headless
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install non-chai dependencies
COPY ./requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

# Install eventlet production server if production build
#RUN if [ "$BUILD_ENV" = "production" ] ; then pip install eventlet ; fi

# Dummy data_dir so things don't break
RUN echo "import os; DATA_DIR=os.path.abspath('.')" >> /risky_overcooked/src/risky_overcooked_rl/data_dir.py

# Copy over remaining files
COPY ./static ./static
COPY ./*.py ./
COPY ./graphics/$GRAPHICS ./static/js/graphics.js
COPY ./config.json ./config.json

# Set environment variables that will be used by app.py
ENV HOST=0.0.0.0
ENV PORT=5000
ENV CONF_PATH=config.json

# Do the thing
EXPOSE 5000
CMD ["python", "-u", "app.py"]




#FROM python:3.11-buster
#
#ARG BUILD_ENV
#ARG OVERCOOKED_BRANCH
#ARG GRAPHICS
#
#WORKDIR /app
#
## Install non-chai dependencies
#COPY ./requirements.txt ./requirements.txt
#RUN pip install -r requirements.txt
#
##ADD ./../../../risky_overcooked /risky_overcooked
#
## Install eventlet production server if production build
#RUN if [ "$BUILD_ENV" = "production" ] ; then pip install eventlet ; fi
#
## Clone chai code
#RUN git clone --recursive https://github.com/mosmith3asu/risky_overcooked.git --branch $OVERCOOKED_BRANCH --single-branch /risky_overcooked
#
## Dummy data_dir so things don't break
#RUN echo "import os; DATA_DIR=os.path.abspath('.')" >> /risky_overcooked/src/risky_overcooked_rl/data_dir.py
#
## Install chai dependencies
##RUN pip install -e '/overcooked_ai[harl]'
#RUN pip install -e '/risky_overcooked[harl]'
#
##RUN apt-get -y update
##RUN apt-get install -y libgl1-mesa-dev
#
## Copy over remaining files
#COPY ./static ./static
#COPY ./*.py ./
#COPY ./graphics/$GRAPHICS ./static/js/graphics.js
#COPY ./config.json ./config.json
#
#
#
## Set environment variables that will be used by app.py
#ENV HOST=0.0.0.0
#ENV PORT=5000
#ENV CONF_PATH=config.json
#
## Do the thing
#EXPOSE 5000
#CMD ["python", "-u", "app.py"]
