# MM-Det Inference Code

This code is originally from [SparkleXFantasy/MM-Det](https://github.com/SparkleXFantasy/MM-Det).  
The test code has been revised to support inference, and additional inference functionality has been added.

## How to Use

### 1. Inference
To perform inference, navigate to the `inference` directory and run the following command:

```bash
cd ./25th-conference-fakebusters/model/MM_Det/
pip install -r requirements.txt
cd ./25th-conference-fakebusters/model/MM_Det/LLaVa/
pip install -e .



cd ./25th-conference-fakebusters/model/MM_Det/inference
python main.py
```

### 2. Check Server

```bash
curl -X POST "[your_address]/process_video/" \
     -H "Content-Type: multipart/form-data" \
     -F "[mp4 folder directory]"
```
