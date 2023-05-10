# pipeline_triplet_extraction
#Preprocessed Dataset

The present directory comprises the preprocessed NCG dataset in csv format for all three tasks, as well as the Scaffold dataset and an additional dataset for the phrase extraction model : https://drive.google.com/drive/folders/1S2ywPcOF0QPxeGlrQ0XhU7Ms4RFq40Rs?usp=sharing.

# Task A

Change the current directory to TaskA

**Heading.ipynb** - To extract Heading and Sub Heading from the ocr file (Training and Trial Dataset)
**Test_Heading.ipynb** - To extract Heading and Sub Heading from the ocr file (Test dataset)
**NCG_Preprocessing.ipynb** - Preprocess the NCG Dataset to csv file

Training the model
**python multitasking-both-scaffold.py**

Test the model
**python test.py**

**Submission.ipynb** - Submission folder generated for codalab submission

# Task B

Change the current directory to TaskB

**Phrase_Extraction_Processing.ipynb** - Preprocess the NCG Dataset for Phrase extraction
**Phrase_Extraction_Predicate.ipynb** - Preprocess the NCG Dataset for Predicate Classification
**SciClaim_Data_Preprocessing.ipynb** - Preprocess the SciClaim Dataset
**SCIERC_Data_Preprocessing.ipynb** - Preprocess the SciERC Dataset
**TaskB_Submission.ipynb** - Submission folder generated for codalab submission

Phrase Extraction
Training - **python train.py**
Test - **python test.py**

# Task C

Change the current directory to TaskC

Predicate Classification 
Training - **python predicate_classification.py**
Test - **python predict_predicate.py ../Preprocessed_Dataset/Test_Predicate.csv predicate_classification.pt**

IU Classification Data Preprocessing Files
**IU_Paragraph_Preprocessing.ipynb**

Triplets Data Preprocessing Files
**Triplets_*_Preprocessing.ipynb** where * stands for type A,B,C,D or NCG

Training IU classification model
**python info_units.py**
**python hyper_setup.py**

Test IU Classification model
**python predict_info_units.py ../Preprocessed_Dataset/Test_IU.csv Test_IU.csv scibert_8_class.pt scibert_hyper_setup.pt**


Training the respective triplets model
**python training_a.py**
**python training_b.py**
**python training_c.py**
**python training_d.py**


Test the respective triplets model
**python test_a.py**
**python test_b.py**
**python test_c.py** 
**python test_d.py**

Prediction of the respective triplets model (Ground truth not available for pipeline prediction)
**python predict_a.py ../Preprocessed_Dataset/Test_Triplets_A.csv scibert_triplets_A.pt**
**python predict_b.py ../Preprocessed_Dataset/Test_Triplets_B.csv scibert_triplets_B.pt**
**python predict_c.py ../Preprocessed_Dataset/Test_Triplets_C.csv scibert_triplets_C.pt**
**python predict_d.py ../Preprocessed_Dataset/Test_Triplets_D.csv scibert_triplets_D.pt**

Codalab Submission Files
**Triplets_Submission.ipynb** 
**IU_Submission.ipynb**


# Pipeline

Copy all the trained models to Trained_Model directory inside Pipeline directory

Run the executable file
**./ncg.sh**
All commands in **./ncg.sh** have following theme
**python filename --input_files --output_files --saved_model**

