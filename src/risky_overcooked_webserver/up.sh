#!/bin/sh

if [[ $1 = production* ]]; then

    if [[ $2 = *local* ]]; then
      echo "production (local)"
      export BUILD_ENV=production_local


      docker-compose -f docker-compose-local.yml build --no-cache
      docker-compose -f docker-compose-local.yml up --force-recreate -d
    else
      docker-compose build --no-cache
      docker-compose up --force-recreate -d
    fi

else
  if [[ $1 = *local* ]]; then
    echo "development (local)"
    export BUILD_ENV=development_local
    docker-compose -f docker-compose-local.yml up --build

  else
    echo "development"
    export BUILD_ENV=development
    # Uncomment the following line if there has been an update to overcooked-ai code
    # docker-compose build --no-cache

    # Force re-build of all images but allow use of build cache if possible
    docker-compose up --build
  fi
fi