import transformers

seed_val = 100
MAX_LEN = 256
TRAIN_BATCH_SIZE = 1
VALID_BATCH_SIZE = 1
TEST_BATCH_SIZE = 1
EPOCHS = 5
BASE_MODEL_PATH = "allenai/scibert_scivocab_uncased"
TESTING_FILE = "../Preprocessed_Dataset/test.csv"
MODEL_PATH = "scibert_5.pt"
TOKENIZER = transformers.BertTokenizer.from_pretrained(
    BASE_MODEL_PATH,
    do_lower_case=True
)
