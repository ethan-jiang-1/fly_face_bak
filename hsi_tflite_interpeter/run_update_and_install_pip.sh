#/usr/bin bash

if [[ -d "../tensorflow/lite/tools/pip_package/gen/tflite_pip/python3" ]]; then
    echo "find compied pip."
else
    echo "no compiled pip found at ./tensorflow/lite/tools/pip_package/gen/tflite_pip/python3"
    echo "please make sure you have build a new pip"
    exit 1
fi

rm -rf tflite_custom/tflite_runtime/__pycache__

echo "clean up previous tflite_custom"

echo "uninstall tflite-runtime and tflite-runtime-inp"
pip uninstall -y tflite-runtime-inp 
pip uninstall -y tflite-runtime
echo "tflite pip after uninstall"
pip list | grep tflite
echo ""

echo "copy latest built pip for "$(uname)

if [ "$(uname)" == "Darwin" ]; then
    sudo rm -rf tflite_custom.macos
    sudo cp -r ../tensorflow/lite/tools/pip_package/gen/tflite_pip/python3 .
    sudo mv python3 tflite_custom.macos
elif [ "$(expr substr $(uname -s) 1 5)" == "Linux" ]; then
    sudo rm -rf tflite_custom.linux

    sudo cp -r ../tensorflow/lite/tools/pip_package/gen/tflite_pip/python3 .
    sudo mv python3 tflite_custom.linux
fi

echo "copy pip done"

./run_install_pip.sh

echo "done"
