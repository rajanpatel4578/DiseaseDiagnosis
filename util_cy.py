from datetime import datetime
import sent2vec 
import os
from nltk import word_tokenize
from nltk.corpus import stopwords
#nltk.download('punkt')
from string import punctuation
from scipy.spatial import distance
from utils import Constants
import numpy as np


def preprocess_diagnosis(diagnosis):
    diagnosis = diagnosis.lower()
    diagnosis = diagnosis.rstrip()
    diagnosis_list = diagnosis.split('--')
    diagnosis_list = list(set(diagnosis_list))
    diagnosis_no_drgtype = list()
    # cdef str d, x
    for d in diagnosis_list:
        d = d.replace("apr:", "")
        d = d.replace("hcfa:", "")
        d = d.replace("ms:", "")
        diagnosis_no_drgtype.append(d)
    diagnosis_no_drgtype = list(set(diagnosis_no_drgtype))
    diagnosis_final = list()

    for x in diagnosis_no_drgtype:
        prefix = ''
        for d in diagnosis_list:
            if x in d:
                prefix = prefix + d[0:d.index(':')] + ","
        prefix = prefix[:-1]
        x = prefix + ":" + x
        diagnosis_final.append(x.rstrip())

    return diagnosis_final


def load_model():
    # dd/mm/YY H:M:S
    start_time = current_time()
    print("LOAD MODEL Start time: ", start_time)
    model = sent2vec.Sent2vecModel()
    model.load_model(os.getcwd() + '/BioSentVec_PubMed_MIMICIII-bigram_d700.bin')
    #model.load_model(os.getcwd() + '/model.bin')
    end_time = current_time()             # <<<<<<<<<<<<<<
    print("LOAD MODEL End time: ", end_time)
    time_diff = elapsed_time(start_time, end_time)             # <<<<<<<<<<<<<<
    print("LOAD MODEL Execution time: " + str(time_diff))
    return model

def current_time():             # <<<<<<<<<<<<<<
    now = datetime.now()
    return now.strftime("%d/%m/%Y %H:%M:%S")

def elapsed_time(start_time, end_time):             # <<<<<<<<<<<<<<
    return datetime.strptime(end_time, Constants.FMT) - datetime.strptime(start_time, Constants.FMT)

def embending_symptoms(model, admissions):
    start_time = current_time()
    print("EMBENDING SYMPTOMS Start time: ", start_time)             # <<<<<<<<<<<<<<
    embending_symptoms = dict()
    for key in admissions:
        a = admissions.get(key)
        symptoms_list = a.symptoms.split(',')             # <<<<<<<<<<<<<<
        for s in symptoms_list:
            symptoms = preprocess_sentence(s)
            embs = model.embed_sentence(symptoms)
            embending_symptoms.update({symptoms:embs})             # <<<<<<<<<<<<<<
    end_time = current_time()
    print("EMBENDING SYMPTOMS End time: ", end_time)
    time_diff = elapsed_time(start_time, end_time)             # <<<<<<<<<<<<<<
    print("EMBENDING SYMPTOMS Execution time: " + str(time_diff))
    return embending_symptoms
    

# nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
def preprocess_sentence(text):
    text = text.replace('/', ' / ')
    text = text.replace('.-', ' .- ')             # <<<<<<<<<<<<<<
    text = text.replace('.', ' . ')
    text = text.replace('\'', ' \' ')
    text = text.lower()             # <<<<<<<<<<<<<<

    tokens = [token for token in word_tokenize(text) if token not in punctuation and token not in stop_words]
    return ' '.join(tokens)


def embending_diagnosis(model, admissions):
    start_time = current_time()
    print("EMBENDING DIAGNOSIS Start time: ", start_time)             # <<<<<<<<<<<<<<
    embending_diagnosis = dict()
    for key in admissions:
        a = admissions.get(key)
        diagnosis_list = a.diagnosis
        for d in diagnosis_list:             # <<<<<<<<<<<<<<
            diagnosis_description = d[d.index(':') + 1:len(d)]
            embs = model.embed_sentence(preprocess_sentence(diagnosis_description))
            embending_diagnosis.update({diagnosis_description:embs})
    end_time = current_time()
    print("EMBENDING DIAGNOSIS End time: ", end_time)             # <<<<<<<<<<<<<<
    time_diff = elapsed_time(start_time, end_time)
    print("EMBENDING DIAGNOSIS Execution time: " + str(time_diff))
    return embending_diagnosis


def init_performance_matrix():
    confusion_matrix = dict()
    for i in {1,0.9,0.8,0.7,0.6}:             # <<<<<<<<<<<<<<
        confusion_matrix.update({i:[0 for x in range(6)]})
    return confusion_matrix


def load_dataset(nFold, name):
    dataset_dir = os.getcwd() + '/Dataset/Fold' + str(nFold) + "/"
    dataset_name = dataset_dir + name             # <<<<<<<<<<<<<<
    dataset_file = open(dataset_name, "r")
    dataset = list()
    for line in dataset_file:
        index = line[0:line.index("_")]
        symptoms = line[line.index("_") + 1: len(line) - 1]             # <<<<<<<<<<<<<<
        symptoms_list = symptoms.split(',')
        symptoms_list_preproc = list()
        for s in symptoms_list:
            symptoms_list_preproc.append(preprocess_sentence(s))             # <<<<<<<<<<<<<<
        # print("****** symptoms_list_preproc:",symptoms)
        dataset.append({index: symptoms_list_preproc})
    return dataset


def init_confusion_matrix():
    confusion_matrix = dict()
    for i in {1,0.9,0.8,0.7,0.6}:             # <<<<<<<<<<<<<<
        confusion_matrix.update({i:[0 for x in range(2)]})
    return confusion_matrix


def predictS2V(i, index, test_admission, test_symptoms, x_train, nrow, ncol, embendings_symptoms, embendings_diagnosis, admissions, similarity_matrix,             # <<<<<<<<<<<<<<
            confusion_matrix_mean_max, confusion_matrix_max, confusion_matrix_Top_K_mean_max_dict, confusion_matrix_Top_K_max_dict, directory_prediction,
            directory_prediction_details, performance_out_file):
    # OPEN PREDICTION OUTPUT FILE
    prediction_out_file = open(directory_prediction + test_admission.hadm_id + '.txt', 'w')             # <<<<<<<<<<<<<<
    detailed_out_file = open(directory_prediction_details + test_admission.hadm_id + '.txt', 'w')
    # START SINGLE PREDICTION TIME
    start_time = current_time()             # <<<<<<<<<<<<<<
    #print_log(prediction_out_file,"PREDICTION Start time: " + start_time + "\n", LOG)
    #print_log(prediction_out_file,"List of TEST Symptoms: " + str(index) + ": " + str(test_symptoms) + "\n", LOG)
    # print("List of TEST Symptoms: " + str(index) + ": " + str(test_symptoms))
    for j in range(0, ncol):
        # TRAIN SYMPTHOMS
        train_symptoms = list(x_train[j].values())[0]             # <<<<<<<<<<<<<<
        #print_log(detailed_out_file,"List of TRAIN Symptoms: " + str(train_symptoms) + "\n", NO_LOG)
        max_symptoms_similarity = dict()
        # COMPUTE MAX SIMILARITY FOR EACH PAIR OF SYMPTOMS
        for x in test_symptoms:             # <<<<<<<<<<<<<<
            #print_log(detailed_out_file,"TEST SYMPTOM : " + x + "\n", NO_LOG)
            test_emb = embendings_symptoms.get(x)
            max_similarity = Constants.MIN_SIMILARITY
            max_symptom = None
            for y in range(0, len(train_symptoms)):             # <<<<<<<<<<<<<<
                #print_log(detailed_out_file,"\t" + train_symptoms[y] + "\t", NO_LOG)
                train_emb = embendings_symptoms.get(train_symptoms[y])
                # similarity = cosine_similarity(test_emb[0], train_emb[0])
                similarity = 1 - distance.cosine(test_emb[0], train_emb[0])
                #print_log(detailed_out_file,str(similarity) + "\n", NO_LOG)
                if similarity > max_similarity:             # <<<<<<<<<<<<<<
                    max_similarity = similarity
                    max_symptom = train_symptoms[y]
            if max_symptom != None:
                max_symptoms_similarity.update({max_symptom + " for " + x: max_similarity})             # <<<<<<<<<<<<<<
                #print_log(detailed_out_file,"Max Similarity Symp : " + max_symptom + " for " + x + "\t" + str(max_similarity) + "\n", NO_LOG)
            else:
                 max_symptoms_similarity.update({"No Similar sympthom for " + x: max_similarity})             # <<<<<<<<<<<<<<
                 #print_log(detailed_out_file,"Max Similarity Symp : " + "No Similar sympthom for " + x + "\t" + str(max_similarity) + "\n", NO_LOG)
        #print_log(detailed_out_file,"MAX SYMPTOMS SIMILARITY : " + str(max_symptoms_similarity) + "\n", NO_LOG)
        # COMPUTE MEAN SIMILARITY THAT WILL BE THE TRAIN ADMISSION SIMILARITY FROM TEST ADMISSION
        min_den = len(test_symptoms)             # <<<<<<<<<<<<<<
        max_den = len(train_symptoms)
        if min_den > max_den:
            max_den = min_den
            min_den = len(train_symptoms)
        mean = 0
        for key in max_symptoms_similarity:
            mean = mean + float(max_symptoms_similarity.get(key))
        mean = mean / max_den
        #print_log(detailed_out_file,"Mean of max similarity: " + str(mean) + "\n\n", NO_LOG)
        similarity_matrix[i, j] = mean
    # print("** similarity_matrix:",similarity_matrix)
    # MAX SIMILARITY
    ###################################################################################
    max = Constants.MIN_SIMILARITY             # <<<<<<<<<<<<<<
    max_index = -1
    #print_log(detailed_out_file,"\nList of TRAIN with DIST <= : " + str(PRUNING_SIMILARITY) + "\n", NO_LOG)
    for j in range(0, ncol):
        #if i != j and similarity_matrix[i, j] >= PRUNING_SIMILARITY:
            #print_log(prediction_out_file,str(similarity_matrix[i, j]) + "\t" + str(list(x_train[j].keys())[0]) + ": " + str( list(x_train[j].values())[0]) + "\n", LOG)
            #print_log(detailed_out_file,str(similarity_matrix[i, j]) + "\t" + str(list(x_train[j].keys())[0]) + ": " + str(list(x_train[j].values())[0]) + "\n", NO_LOG)
        if similarity_matrix[i, j] >= Constants.PRUNING_SIMILARITY and similarity_matrix[i, j] > max:
            max = similarity_matrix[i, j]             # <<<<<<<<<<<<<<
            max_index = j
    gt_diagnosis = test_admission.diagnosis
    print("Max similarity of Symptoms = " + str(max))
    print("List of Symptoms to predict: " + str(index) + ": " + str(test_symptoms))
    print("GT diagnosis: " + str(gt_diagnosis))
    if max_index != -1:             # <<<<<<<<<<<<<<
        most_similar_index = list(x_train[max_index].keys())[0]
        most_similar_symptoms = list(x_train[max_index].values())[0]
        #print_log(prediction_out_file,"Similar List of symptoms: " + str(most_similar_index) + ": " + str(most_similar_symptoms) + "\n", LOG)
        most_similar_admission = admissions.get(most_similar_index)
        predicted_diagnosis = most_similar_admission.diagnosis
        diagnosis_similarity_max = get_diagnosis_similarity_by_description_max(embendings_diagnosis, gt_diagnosis, predicted_diagnosis, 'cosine')
        print("Predicted diagnosis: " + str(predicted_diagnosis))
        print("Diagnosis Similarity Max: " + str(diagnosis_similarity_max))
        # UPDATE CONFUSION MATRIX
        for b in {1, 0.9, 0.8, 0.7, 0.6}:             # <<<<<<<<<<<<<<
            """
            values = confusion_matrix_mean_max.get(b)
                values[FP] += 1
            """
            values = confusion_matrix_max.get(b)             # <<<<<<<<<<<<<<
            if diagnosis_similarity_max >= b:
                values[Constants.TP] += 1
            else:
                values[Constants.FP] += 1
    #else:
        #print_log(prediction_out_file,"Similar List of symptoms: --- \n", LOG)
    # TOP-K SIMILARITY
    ###################################################################################
    similarity_array = np.array(similarity_matrix[i,])             # <<<<<<<<<<<<<<
    largest_indices = np.argsort(-1 * similarity_array)[:(Constants.TOP_K_UPPER_BOUND - Constants.TOP_K_INCR)]
    top_k_admission = list()
    #print_log(prediction_out_file,"TOP-" + str((TOP_K_UPPER_BOUND - TOP_K_INCR)) + ":\n", LOG)
    #print_log(detailed_out_file,"TOP-" + str((TOP_K_UPPER_BOUND - TOP_K_INCR)) + ":\n", NO_LOG)
    top_similarities_max = list()             # <<<<<<<<<<<<<<
    index_top = 0
    for top_index in largest_indices:
        if similarity_matrix[i, int(top_index)] >= Constants.PRUNING_SIMILARITY:             # <<<<<<<<<<<<<<
            most_similar_index = list(x_train[top_index].keys())[0]
            most_similar_symptoms = list(x_train[top_index].values())[0]
            most_similar_admission = admissions.get(most_similar_index)
            predicted_diagnosis = most_similar_admission.diagnosis
            diagnosis_similarity_max = get_diagnosis_similarity_by_description_max(embendings_diagnosis, gt_diagnosis, predicted_diagnosis, 'cosine')
            top_similarities_max.append(diagnosis_similarity_max)
            #print_log(prediction_out_file,str(index_top) + "\t Similar List of symptoms: " + str(most_similar_index) + ": " + str(most_similar_symptoms) + "\n", LOG)
            #print_log(prediction_out_file,str(index_top) + "\t Predicted diagnosis: " + str(predicted_diagnosis) + "\n", LOG)
            #print_log(detailed_out_file,str(index_top) + "\t Predicted diagnosis: " + str(predicted_diagnosis) + "\n", NO_LOG)
            #print_log(detailed_out_file,str(index_top) + "\t Diagnosis Similarity Max: " + str(diagnosis_similarity_max) + "\n", NO_LOG)
            index_top += 1             # <<<<<<<<<<<<<<
    # UPDATE CONFUSION MATRIX TOP K
    for topk in range(Constants.TOP_K_LOWER_BOUND, Constants.TOP_K_UPPER_BOUND, Constants.TOP_K_INCR):
        confusion_matrix_Top_K_max = confusion_matrix_Top_K_max_dict.get(topk)
        for b in {1, 0.9, 0.8, 0.7, 0.6}:
            values = confusion_matrix_Top_K_max.get(b)
            if containGreaterOrEqualsValue(topk, top_similarities_max, b):
                values[Constants.TP] += 1
            else:
                if len(top_similarities_max) > 0:
                    values[Constants.FP] += 1

    end_time = current_time()             # <<<<<<<<<<<<<<
    time_diff = elapsed_time(start_time, end_time)
    #print_log(prediction_out_file,"PREDICTION End time: " + end_time + "\n", LOG)
    #print_log(prediction_out_file,"PREDICTION Execution time: " + str(time_diff) + "\n", LOG)
    # CLOSE PREDICTION OUTPUT FILE
    prediction_out_file.close()             # <<<<<<<<<<<<<<
    detailed_out_file.close()
    # COMPUTE PERFORMANCE INDEX
    performance_out_file.write(str(i) + " - HADM_ID=" + str(test_admission.hadm_id) + ": PERFORMANCE INDEX of MAX SIMILARITY by MAX" + "\n")
    compute_performance_index(confusion_matrix_max, nrow, performance_out_file)             # <<<<<<<<<<<<<<
    for topk in range(Constants.TOP_K_LOWER_BOUND, Constants.TOP_K_UPPER_BOUND, Constants.TOP_K_INCR):
        confusion_matrix_Top_K_max = confusion_matrix_Top_K_max_dict.get(topk)
        performance_out_file.write(str(i) + " - HADM_ID=" + str(test_admission.hadm_id) + ": PERFORMANCE INDEX of TOP-" + str(topk) + " SIMILARITY by MAX" + "\n")
        compute_performance_index(confusion_matrix_Top_K_max, nrow, performance_out_file)             # <<<<<<<<<<<<<<
    print("END PREDICTION " + test_admission.hadm_id + " in " + str(time_diff))




def get_diagnosis_similarity_by_description_max(embendings_diagnosis, gt_diagnosis, predicted_diagnosis, method):             # <<<<<<<<<<<<<<
    MIN_SIMILARITY = 0
    max_diagnosis_similarity = dict()
    max_similarity = MIN_SIMILARITY             # <<<<<<<<<<<<<<
    for x in gt_diagnosis:
        x_description = x[x.index(':')+1:len(x)]
        for y in predicted_diagnosis:
            y_description = y[y.index(':')+1:len(y)]
            emb_diagnosis_to_predict = embendings_diagnosis.get(x_description)
            emb_diagnosis_predicted = embendings_diagnosis.get(y_description)
            # diagnosis_similarity = cosine_similarity(emb_diagnosis_to_predict[0], emb_diagnosis_predicted[0])
            diagnosis_similarity = 1 - distance.cosine(emb_diagnosis_to_predict[0], emb_diagnosis_predicted[0])
            if diagnosis_similarity > max_similarity:
                max_similarity = diagnosis_similarity
    return max_similarity


def containGreaterOrEqualsValue(topK,top_similarities,b):
    for i in range(0,topK):
        if i < len(top_similarities) and top_similarities[i] >= b:             # <<<<<<<<<<<<<<
            return True
    return False


def compute_performance_index(confusion_matrix, nrow, performance_out_file):
    performance_out_file.write(Constants.PERFORMANCE_INDEX_HEADER)             # <<<<<<<<<<<<<<
    for cm in {1, 0.9, 0.8, 0.7, 0.6}:
        values = confusion_matrix.get(cm)
        tp = values[Constants.TP]
        fp = values[Constants.FP]
        if (tp + fp) != 0:
            precision = tp / (tp + fp)
        else:
            precision = 0
        recall = tp / nrow
        if recall + precision != 0:
            f_score = (2 * recall * precision) / (recall + precision)
        else:
            f_score = 0
        prediction_rate = (tp + fp) / nrow
        performance_out_file.write(
            str(cm) + "\t" + str(tp) + "\t" + str(fp) + "\t" + str(precision) + "\t" + str(recall) + "\t" + str(
                f_score) + "\t" + str(prediction_rate) + "\n")
        performance_out_file.flush()

    
def compute_aggregated_performance_index(confusion_matrix, performance_matrix, nrow, performance_out_file):
    performance_out_file.write(Constants.PERFORMANCE_INDEX_HEADER)
    for i in {1, 0.9, 0.8, 0.7, 0.6}:             # <<<<<<<<<<<<<<
        # UPDATE PERFORMANCE INDEX
        confusion_values = confusion_matrix.get(i)
        values = performance_matrix.get(i)
        tp = confusion_values[Constants.TP]
        fp = confusion_values[Constants.FP]
        if (tp + fp) != 0:
            precision = tp / (tp + fp)
        else:
            precision = 0
        recall = tp / nrow
        if recall + precision != 0:
            f_score = (2 * recall * precision) / (recall + precision)
        else:
            f_score = 0
        prediction_rate = (tp + fp) / nrow
        performance_out_file.write(
            str(i) + "\t" + str(tp) + "\t" + str(fp) + "\t" + str(precision) + "\t" + str(recall) + "\t" + str(
                f_score) + "\t" + str(prediction_rate) + "\n")
        performance_out_file.flush()
        # UPDATE PERFORMANCE INDEX
        values[Constants.TP] += tp
        values[Constants.FP] += fp
        values[Constants.P] += precision
        values[Constants.R] += recall
        values[Constants.FS] += f_score
        values[Constants.PR] += prediction_rate


def print_performance_index(performance_matrix, performance_out_file):
    performance_out_file.write(Constants.PERFORMANCE_INDEX_HEADER)
    for i in {1, 0.9, 0.8, 0.7, 0.6}:             # <<<<<<<<<<<<<<
        values = performance_matrix.get(i)
        performance_out_file.write(
            str(i) + "\t" + str(values[Constants.TP] / Constants.K_FOLD) + "\t" + str(values[Constants.FP] / Constants.K_FOLD) + "\t" + str(
                values[Constants.P] / Constants.K_FOLD) + "\t" + str(values[Constants.R] / Constants.K_FOLD) + \
            "\t" + str(values[Constants.FS] / Constants.K_FOLD) + "\t" + str(values[Constants.PR] / Constants.K_FOLD) + "\n")