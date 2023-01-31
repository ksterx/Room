#!/bin/bash

if [ "$(uname)" == 'Darwin' ]; then
    open http://127.0.0.1:5000/
elif [ "$(expr substr $(uname -s) 1 5)" == 'Linux' ]; then
    xdg-open http://127.0.0.1:5000/
fi

mlflow server --backend-store-uri file:./experiments/results --host 127.0.0.1
