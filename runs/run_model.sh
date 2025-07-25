#!/bin/bash

# example usage:
#   ./run_model.sh -f=yolov5 -v=n -s=320 -p=u250

# argument parser
for i in "$@"; do
  case $i in
    -f=*|--family=*)
      FAMILY="${i#*=}"
      shift # past argument=value
      ;;
    -v=*|--variant=*)
      VARIANT="${i#*=}"
      shift # past argument=value
      ;;
    -s=*|--size=*)
      SIZE="${i#*=}"
      shift # past argument=value
      ;;
    -p=*|--platform=*)
      PLATFORM="${i#*=}"
      shift # past argument=value
      ;;
    -*|--*)
      echo "Unknown option $i"
      exit 1
      ;;
    *)
      ;;
  esac
done

# setup strings and paths
ID=${FAMILY}${VARIANT}-${SIZE}-${PLATFORM}
MODEL_NAME=${FAMILY}${VARIANT}_imgsz${SIZE}_fp16
MODEL_PATH=../onnx_models/${FAMILY}/${MODEL_NAME}.onnx
FPGACONVNET_MODEL_PATH=../onnx_models/${FAMILY}/${MODEL_NAME}-fpgaconvnet.onnx
PLATFORM_PATH=../platforms/${PLATFORM}.toml
OUTPUT_PATH=${ID}

## make the output directory
mkdir -p $OUTPUT_PATH

## perform preprocessing
python3 ${FAMILY}-preprocess.py $MODEL_PATH $FPGACONVNET_MODEL_PATH $SIZE

## optimise the model
python3 optimise.py $FPGACONVNET_MODEL_PATH $PLATFORM_PATH $OUTPUT_PATH

## post process the config to make it suitable for hardware generation
python3 ${FAMILY}-postprocess.py $OUTPUT_PATH/config.json $OUTPUT_PATH

