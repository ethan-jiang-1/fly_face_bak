#/usr/bin bash

rm -rf tflite_custom/tflite_runtime/__pycache__

export PROJECT_NAME="tflite-runtime-hsi"
export PACKAGE_VERSION="2.9.04"

pip uninstall tflite-runtime
pip uninstall tflite-runtime-inp
pip uninstall tflite-runtime-hsi

echo "tflite pip install in linux"
pip install -e tflite_custom.linux    

echo "tflite pip after reinstall"
pip list | grep tflite
echo ""

echo "install pip done"
