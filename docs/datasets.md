# Dataset preparation

EdgeCraft accepts a local dataset directory. Keep its labels, annotation files,
and official split definitions alongside the inputs. The agent examines these
files and writes a loader; the source catalog does not select a model or search
strategy.

## A small starting example

With the controller environment installed, download TweetEval sentiment:

```bash
python scripts/prepare_dataset.py tweet_eval --output "$DATA_ROOT/tweet_eval"
edgecraft data inspect "$DATA_ROOT/tweet_eval"
```

The helper supports the Hugging Face datasets listed by `--help` and official
CIFAR-10 Python batches. It refuses to overwrite an existing destination. For
other datasets, follow the source link below, unpack the data, and pass that
directory to `edgecraft synth --dataset`. You may use your own directory names.

For Kaggle-hosted datasets, first accept any dataset or competition terms and
configure the Kaggle CLI locally, then use `kaggle datasets download -d OWNER/SLUG
--unzip -p DESTINATION`, or `kaggle competitions download -c COMPETITION -p
DESTINATION` followed by extraction. Credentials stay in your account's local
configuration. For datasets requiring registration, obtain access through the
original provider. EdgeCraft does not redistribute third-party data.

## Public benchmark inputs

Task IDs follow Table 4. Sources and suggested local directory names are also
available in `edgecraft/config/datasets.json` and `edgecraft data list`.

| Task | Dataset / source | Device | Expected inputs |
| --- | --- | --- | --- |
| T01 | [GlobalWheat](https://zenodo.org/records/4271191) | NX | Images and COCO JSON boxes |
| T02 | [HomeObjects-3K](https://docs.ultralytics.com/datasets/detect/homeobjects-3k/) | NX | YOLO images, labels, and YAML |
| T03 | [KITTI](https://www.cvlibs.net/datasets/kitti/eval_object.php) | Orin | image_2/ and label_2/ |
| T04 | [DOTA-v1.0](https://captain-whu.github.io/DOTA/dataset.html) | Orin | images/ and labelTxt/ |
| T05 | [PASCAL VOC](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/) | NX | JPEGImages/, Annotations/, and ImageSets/ |
| T06 | [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html) | TX2 | Official Python batches |
| T07 | [Caltech-101](https://data.caltech.edu/records/mzrjq-6wc02) | TX2 | 101_ObjectCategories/ class folders |
| T08 | [NEU-CLS](http://faculty.neu.edu.cn/songkc/en/zdylm/263265/list/index.htm) | TX2 | Class folders or image-label lists |
| T09 | [GTSRB](https://benchmark.ini.rub.de/gtsrb_dataset.html) | TX2 | Class folders and CSV labels |
| T10 | [Crack Segmentation](https://docs.ultralytics.com/datasets/segment/crack-seg/) | NX | Images and segmentation labels |
| T11 | [Dog-Pose](https://docs.ultralytics.com/datasets/pose/dog-pose/) | NX | Images and pose keypoints |
| T12 | [Hand Keypoints](https://docs.ultralytics.com/datasets/pose/hand-keypoints/) | NX | Images and hand keypoints |
| T13 | [VisDrone](https://github.com/VisDrone/VisDrone-Dataset) | Orin | Images and detection annotations |
| T14 | [Adience](https://talhassner.github.io/home/projects/Adience/Adience-data.html) | TX2 | Aligned faces and fold text files |
| T15 | [Rendered SST2](https://github.com/openai/CLIP/blob/main/data/rendered-sst2.md) | TX2 | Images and sentiment labels (Parquet or DatasetDict) |
| T16 | [VeRi-776](https://vehiclereid.github.io/VeRi/) | Orin | image_train/, image_test/, and identity annotations |
| T17 | [ShanghaiTech](https://www.crowd-counting.com/) | Orin | Images and ground-truth MAT files |
| T18 | [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad) | NX | Category/train, test, and ground_truth |
| T19 | [Loghub-BGL-2K](https://github.com/logpai/loghub/tree/master/BGL) | PC | BGL_2k.log_structured.csv with anomaly labels |
| T20 | [Loghub-HDFS](https://github.com/logpai/loghub/tree/master/HDFS) | PC | Log records and anomaly labels; or labeled Parquet |
| T21 | [CLINC150](https://github.com/clinc/oos-eval) | PC | data_full.json or split CSV files |
| T22 | [BANKING77](https://github.com/PolyAI-LDN/task-specific-datasets/tree/master/banking_data) | Pi 5 | Split CSV files with text and category |
| T23 | [AG News](https://huggingface.co/datasets/fancyzhx/ag_news) | NX | Split text/label tables |
| T24 | [Ecommerce Text](https://www.kaggle.com/datasets/saurabhshahane/ecommerce-text-classification) | PC | Category and description CSV |
| T25 | [WebMD Reviews](https://www.kaggle.com/datasets/rohanharode07/webmd-drug-reviews-dataset) | PC | Review text and rating columns |
| T26 | [Feedback Prize ELL](https://www.kaggle.com/competitions/feedback-prize-english-language-learning/data) | PC | Text and six score columns |
| T27 | [Airline Reviews](https://www.kaggle.com/datasets/juhibhojani/airline-reviews) | PC | Review text and rating columns |
| T28 | [GLUE STS-B](https://huggingface.co/datasets/nyu-mll/glue) | PC | Train/dev TSV with sentence pairs and scores |
| T29 | [SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) | Pi 5 | SMSSpamCollection (tab-delimited label/text) |
| T30 | [TweetEval](https://github.com/cardiffnlp/tweeteval) | Pi 5 | Sentiment train/validation/test splits |
| T31 | [SST-2](https://dl.fbaipublicfiles.com/glue/data/SST-2.zip) | Pi 5 | Sentence and binary sentiment labels |
| T32 | [GoEmotions](https://github.com/google-research/google-research/tree/master/goemotions) | TX2 | Text and emotion annotations |
| T33 | [BoolQ](https://github.com/google-research-datasets/boolean-questions) | NX | Question, passage, and answer |
| T34 | [LibriSpeech-100h](https://www.openslr.org/12) | Orin | LibriSpeech audio and .trans.txt transcripts |
| T35 | [Fluent Speech Commands](https://zenodo.org/records/11106540) | TX2 | WAV files and intent annotations; or DatasetDict |
| T36 | [SLURP](https://github.com/pswietojanski/slurp) | NX | Audio and JSONL annotations; or Parquet |
| T37 | [RAVDESS Audio-Speech](https://zenodo.org/records/1188976) | NX | Actor_*/ speech WAV files with filename labels |
| T38 | [ESC-50](https://github.com/karolpiczak/ESC-50) | TX2 | audio/ and meta/esc50.csv; or DatasetDict |
| T39 | [Speech Commands](http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz) | Pi 5 | Class directories and validation/testing lists |
| T40 | [HHAR](https://archive.ics.uci.edu/dataset/344/heterogeneity+activity+recognition) | Pi 5 | Sensor CSV files and activity labels |
| T41 | [MotionSense](https://github.com/mmalekzadeh/motion-sense) | Pi 5 | Activity/trial directories of sensor CSV files |
| T42 | [Ethanol Concentration](https://www.timeseriesclassification.com/description.php?Dataset=EthanolConcentration) | Pi 5 | TRAIN/TEST .ts or .arff files |
| T43 | [Self-Regulation-SCP1](https://www.timeseriesclassification.com/description.php?Dataset=SelfRegulationSCP1) | Pi 5 | TRAIN/TEST .ts or .arff files |
| T44 | [Heartbeat](https://www.timeseriesclassification.com/description.php?Dataset=Heartbeat) | Pi 5 | TRAIN/TEST .ts or .arff files |
| T45 | [Banana Quality](https://www.kaggle.com/datasets/mrmars1010/banana-quality-dataset) | PC | Feature and Quality CSV columns |
| T46 | [Software Defects](https://www.kaggle.com/datasets/semustafacevik/software-defect-prediction) | PC | Feature and defect-label table |
| T47 | [UCI Abalone](https://archive.ics.uci.edu/dataset/1/abalone) | PC | abalone.data; last column is Rings |
| T48 | [Wild Blueberry Yield](https://www.kaggle.com/datasets/saurabhshahane/wild-blueberry-yield-prediction) | PC | Feature and yield CSV columns |
| T49 | [Clotho](https://zenodo.org/records/4783391) | Orin | Audio splits and caption CSV files |
| T50 | [Flickr8k](https://huggingface.co/datasets/jxie/flickr8k) | Orin | Images and captions; or image/caption Parquet |

## Splits and task details

Use the provided train/validation/test splits for model selection and final
evaluation. For a single labeled table, specify the target column and desired
split policy in the request. For subject-based or time-based datasets, state the
grouping constraint explicitly. The loader must preserve these boundaries.

The preflight reports sample formats, shapes, label hints, and split evidence;
it does not establish model quality. Keep the original annotation semantics:
for example, Speech Commands uses directory labels, RAVDESS encodes attributes
in filenames, and captioning datasets pair media with text. For multi-label
tasks, explain whether the desired output is multi-label or a restricted
single-label subset. Include the desired metric and SLOs in your task request.

`edgecraft data inspect DIRECTORY` reads a few samples locally without calling
an LLM. Full synthesis additionally sends selected dataset observations to the
configured provider; see [Security](../SECURITY.md#data-sent-to-an-llm-provider).
