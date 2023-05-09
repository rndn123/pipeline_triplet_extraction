#python TaskA_Sentences/predict_sentences.py NCG_Dataset/NCG_Dataset.csv NCG_Dataset/NCG_sentences.csv Trained_Model/model_256.pt
#python TaskB_Phrases/predict_phrase_ensemble.py NCG_Dataset/NCG_sentences.csv NCG_Dataset/NCG_phrases_ensemble.csv Trained_Model/ensemble_models
#python TaskB_Phrases/phrase_ensembler.py NCG_Dataset/NCG_phrases_ensemble.csv NCG_Dataset/NCG_phrases.csv
python TaskB_Phrases/predict_phrase.py NCG_Dataset/ScienceQA_questions.csv NCG_Dataset/NCG_phrases.csv Trained_Model/scibert_crf_f1.pt
python TaskC_Info_Units/predict_info_units.py NCG_Dataset/ScienceQA_questions.csv NCG_Dataset/NCG_info_units.csv Trained_Model/scibert_info_units_8.pt Trained_Model/scibert_hyper_setup.pt
python TaskC_Predicate/predicate_preprocessing.py NCG_Dataset/NCG_phrases.csv NCG_Dataset/NCG_predicate.csv
python TaskC_Predicate/predict_predicate.py NCG_Dataset/NCG_predicate.csv Trained_Model/predicate_classification.pt
python TaskC_Triplets/triplets_preprocessing.py NCG_Dataset/NCG_info_units.csv NCG_Dataset/NCG_predicate.csv NCG_Dataset/NCG_triplets.csv NCG_Dataset/NCG_triplets_a.csv NCG_Dataset/NCG_triplets_b.csv NCG_Dataset/NCG_triplets_c.csv NCG_Dataset/NCG_triplets_d.csv
python TaskC_Triplets/triplets_a.py NCG_Dataset/NCG_triplets_a.csv Trained_Model/scibert_triplets_A.pt
python TaskC_Triplets/triplets_b.py NCG_Dataset/NCG_triplets_b.csv Trained_Model/scibert_triplets_B.pt
python TaskC_Triplets/triplets_c.py NCG_Dataset/NCG_triplets_c.csv Trained_Model/scibert_triplets_C.pt
python TaskC_Triplets/triplets_d.py NCG_Dataset/NCG_triplets_d.csv Trained_Model/scibert_triplets_D.pt
python TaskC_Triplets/triplets_results.py NCG_Dataset/NCG_triplets_a.csv NCG_Dataset/NCG_triplets_b.csv NCG_Dataset/NCG_triplets_c.csv NCG_Dataset/NCG_triplets_d.csv NCG_Dataset/NCG_triplets_results.csv
