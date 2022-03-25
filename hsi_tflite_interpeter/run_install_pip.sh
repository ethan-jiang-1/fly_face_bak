#/usr/bin bash

rm -rf tflite_custom/tflite_runtime/__pycache__

export PROJECT_NAME="tflite-runtime-hsi"
export PACKAGE_VERSION="2.9.03"

pip uninstall tflite-runtime
pip uninstall tflite-runtime-inp
pip uninstall tflite-runtime-hsi

if [ "$(uname)" == "Darwin" ]; then
    pip install -e tflite_custom.macos    
elif [ "$(expr substr $(uname -s) 1 5)" == "Linux" ]; then
    pip install -e tflite_custom.linux    
fi

echo "tflite pip after reinstall"
pip list | grep tflite
echo ""

echo "install pip done"
