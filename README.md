# Fingerprint Classification in Raspberry Pi

This is the code repository of Fingerprint Classification module for running on Edge device of Raspberry Pi. 

## Prerequisites
To install the Python dependencies, run:
```
pip install -r requirements.txt
```

Next, to run the code on Raspberry Pi, use `piFingerprint.py` as follows:

```
python3 piFingerprint.py --filename fake.BMP --model_path fprintmodel1.tflite --label_path labels.txt
```


