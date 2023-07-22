### Setup environment 

## Native training YAMNet in TFLite
Runing TFLite might result in some errors when using newer TF or Python versions. Please follow the instructions carefully to guarantee replicability.

###### With conda env
- To create conda environment: 
```bash
conda create --name yamnet-env python=3.8
conda activate yamnet-env 
```

- To install libraries for audio task: 
```bash
sudo apt-get update
sudo apt-get install libsndfile1 -y
sudo apt-get install ffmpeg -y
```

- To install requirements:
```bash
pip install -r requirements.txt
```
### Dataset

###### Load the data
- To download esc50 dataset:
```bash
python3 main.py --scenario  download_data --dataset_name esc-50
```
- To unzip esc-50 dataset:
```bash
unzip ./dataset-esc50/esc-50.zip
```

- To download nbac dataset:
```bash
python3 main.py --scenario  download_data --dataset_name nbac
```

###### Prepare data

- To create category esc-50 dataset: 
```bash
python3 main.py --scenario  create_category_dataset --dataset_name esc-50
```
- To create category nbac dataset: 
```bash
python3 main.py --scenario  create_category_dataset --dataset_name nbac
```

###### Split dataset 
- To train test split:
```bash
python3 main.py --scenario train_test_split --test_data_ratio 0.2 
```
- Flags: 
	- `--test_data_ratio`: ratio of test and train data

###### Augment dataset
- To augment the train dataset:
```bash
python3 main.py --scenario  augment_dataset
```


### Train and Export

- To run training:
```bash
python3 main.py --scenario train \
--train_data_ratio 0.8 \
--epochs 50 \
--batch_size 32 \
--tflite_file_name model.tflite \
--save_path ./model
```
- Flag:
	- `--train_data_ratio`: ratio of train data and val data
	- `--epochs`: num epochs 
	- `--batch_size`: num batch size 
	- `--tflite_file_name`: the tflite model name
	- `--save_path`: path to directory contains model 

- To check tflite_model_info: 
```bash
python3 main.py --scenario tflite_model_info
```

### Notes: 
- To modify parameters, go to `scenario/args.py` or through command. 
 
### Build APK file: 

###### Install Android Studio 
The desktop app can be installed for Windows or for WSL follow this instructions:
- To install, use this tutorial `https://linuxhint.com/install-android-studio-linux-mint-and-ubuntu/`
- Or run the following commands:
```bash
sudo apt update
sudo apt install openjdk-11-jdk
sudo snap install android-studio –classic
```

###### Run default audio_classification app
- To get the repo `https://github.com/tensorflow/examples/tree/master/lite/examples/audio_classification/android`:
```bash
git clone https://github.com/tensorflow/examples.git
cp ./examples/lites/examples/audio_classification ./
cd audio_classification
```
- Start Android Studio, open the project located in `audio_classification/android`, run app with default model: 
```bash
- Select target device menu.
- Click `Run`.

```
###### Copy the model to assets
- To run with custom model, copy `path/to/model.tflite` to the android app: 
```bash
cp path/to/model.tflite  audio_classification/android/app/src/main/assets/
```

###### Modify params on Android Studio
Go to `/android/app/src/main/java/org/tensorflow/lite/examples/audio/AudioClassificationHelper.kt`. 
- To change model name, at line 136:
```bash
const val YAMNET_MODEL = "path/to/model.tflite"
```
- To change length recordings, (change 1000ms->5000ms), at line 105:
```bash
val lengthInMilliSeconds = ((classifier.requiredInputBufferSize * 1.0f) /
                classifier.requiredTensorAudioFormat.sampleRate) * 5000
```
- To get the result of custom model, change output index from 0->1 (0: result from original yamnet, 1: result from custom yamnet), at line 122: 
```bash
listener.onResult(output[1].categories, inferenceTime)
```

###### Build APK file
- Click `Run` to build the app. In the toolbar, to build the APK file, click `Build>Build Bunder(s)/APK(s)>Build APK(s)
- Get the APK file at `/audio_classification/android/app/build/intermediates/apk/debug`
- Copy the APK file to the android phone and install. 


Author: David Castelló Tejera
