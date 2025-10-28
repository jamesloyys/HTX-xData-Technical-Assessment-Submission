# HTX-xData-Technical-Assessment

To run the code in this repo, you need to set up a Python environment and install the dependencies in the `requirements.txt` file. 

Use conda to create a new virtual environment:

```bash
conda create -n htx python=3.10
conda activate htx
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

You would also need to download the Common Voice dataset and place the files in a `data` folder in the root directory of the repo.

The structure should be:

```
data/
    common_voice/
        cv-valid-dev/
        cv-valid-test/
        cv-valid-train/
        .
        .
        .

asr/
asr-train/
hotword-detection/
README.md
requirements.txt
.gitignore
```

## Usage

To start the docker container for task 1e, run the following commands:

```bash
cd asr
docker build -t asr-api .
docker run -d -p 8001:8001 asr-api
```