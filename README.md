# Transformers based Deep Image Captioning

### Introduction
The project is developed for the thesis submission of Masters work at LJMU.
The project is an image captioning system built using CNN models and transformer decoder.

The file contains the details about the codes and the way to execute the files.

The project contains two sets of code files. One folder 'codes_gpt_embeddings' has the code for building the model using GPT2 embeddings.
The folder 'codes_trained_embeddings' has the code for building the model with trained embedding layer.

### Folder Structure

The following folder structure is followed for both the folders

For 'codes_gpt_embeddings' folder
- dataset
	- Flickr8k_Dataset
		-** All the images in the datatset are in this folder
	- Flickr8k_text
		- CrowdFlowerAnnotations.txt
		- ExpertAnnotations.txt
		- Flickr_8k.devImages.txt
		- Flickr_8k.testImages.txt
		- Flickr_8k.trainImages.txt
		- Flickr8k.lemma.token.txt
		- Flickr8k.token.txt
- model_saves
- experiments
	- transformerTest.csv
	- transformerValidation.csv
- bleu.py
- eval.py
- model.py
- requirements.txt
- train.py
- util.py

For 'codes_trained_embeddings' folder
- dataset
	- Flickr8k_Dataset
		- ** All the images in the datatset are in this folder
	- Flickr8k_text
		- CrowdFlowerAnnotations.txt
		- ExpertAnnotations.txt
		- Flickr_8k.devImages.txt
		- Flickr_8k.testImages.txt
		- Flickr_8k.trainImages.txt
		- Flickr8k.lemma.token.txt
		- Flickr8k.token.txt
- model_saves
- experiments
	- transformerTest.csv
	- transformerValidation.csv
- bleu.py
- eval.py
- model.py
- requirements.txt
- train.py
- util.py
- vocab.txt
- vocab_builder.py

### Explanation of Files
To train and test the models use, the train.py file. 
models.py file contains the models code.
eval.py file contains the evaluation metrics code.
util.py contains some helper functions needed to clean the data.
model_saves folder contains the saved models from any test runs
experiments folder contains the metrics for any test runs.

The 'codes_trained_embeddings' contain two extra files.
vocab.txt file contains all the vocabulary in the captions dataset.
vocab_builder.py contains the code to build the above vocabulary.

The Flickr8k dataset should be present inside the 'dataset' folder as mentioned in the above folder structure.
The Flickr8k images and text zip files can be downloaded from the below links.
https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip 
https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip

### Training and Evaluating the models
To train the model we have to mainly run the train.py model with arguements.
During the thesis implementation, all the codes were executed on google colaboratory. The environment used Python 3.7.

Before running the train.py the requirements present in the requirements.txt need to be installed.
Run the following command to install the libraries.

```
pip install -r requirements.txt
```

If the code from 'codes_trained_embeddings' is being executed then first run the vocab_builder.py to prepare the vocabulary file.

```
python vocab_builder.py
```

The next is common for both the trained embeddings and GPT2 embeddings code. Before that the list of arguments to run the codes are presented below

```
--lr -> Learning rate | Default=0.0001
--batch-size -> Batch Size for training dataset | Default=64
--batch-size-val ->  Batch Size for validation dataset | Default=64
--encoder-type -> CNN architecture to use for the model | Default='resnet18' | Available Options=['resnet18', 'resnet50', 'resnet101','resnext101_64x4d','seresnext101_64x4d','resnesta269','resnesta200','efficientnet_b4b','efficientnet_b3b', efficientnet_b7b]
--fine-tune -> Fine Tune the CNN model or not | Default=0 | Available Options = [0,1] | 0->No fine tuning, 1->Apply Fine Tune
--beam-width -> Beam widht during model evaluation | Default=4
--num-epochs -> Numbers of epochs to train the model for | Default = 100
--decoder-hidden-size -> Hidden size length in the decoder | Default = 512
--experiment-name -> Name given to the experiment being conducted 
--num-tf-layers -> Number of layers in the decoder transformer | Default = 3
--num-heads -> Number of heads in the decoder transformer | Default = 2
--beta1 -> beta1 value in the Adam optimizer | Default = 0.9
--beta2 -> Beta2 value in the Adam optimizer | Default = 0.999
--dropout-trans -> Drop out value in the decoder transformer | Default = 0.1
--smoothing -> Whether to apply label smoothing or not | Default =1 | Available options = [0,1] | 0->No label smoothing, 1->Apply Label Smoothing
--Lepsilon -> Label smoothing Lepsilon value | Default=0.1
--use-checkpoint -> Whether to load a previously save model or not | Default=0| Available options=[0,1] | 0->Don't use checkpoint, 1->Use checkpoint
--img-resize -> The value to which input image should be resize | Default=22
--mode -> Whether the model should be in train or test mode. If in test mode, the file loads the saved model by name experiment name from the model_saves folder. | Default=train | Available options: ['train','test']
```

Below is a sample code to run the train.py

```
 !python train.py --lr 0.00004 --encoder-type resnet18 --batch-size 16 --batch-size-val 16 --num-epochs 20 --num-heads 1 --num-tf-layers 1 --dropout-trans 0.1 --img-resize 300 --beam-width 6  --experiment-name resnext101_bs64_ft1_l3_h2_bw6_bert --fine-tune 1 --smoothing 1
```

