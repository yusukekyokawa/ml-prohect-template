#!/bin/sh
cat << "EOF"
ML FLOW UI!
EOF
#MLFLOW_VERSION=$(mlflow --version)
# echo "Ml Flow Server '$MLFLOW_VERSION' - Python: $PYTHON_VERSION"

#    --backend-store-uri ${PATH_TO_TRACKING}/myruns \
exec mlflow server \
    --backend-store-uri ../mlruns \
    --default-artifact-root ../artifact \
    --host 0.0.0.0