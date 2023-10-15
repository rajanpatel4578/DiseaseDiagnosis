import os
import shutil
import sys
from math import floor

from gensim.models import fasttext

import numpy
import numpy as np
import sent2vec
import sklearn
from scipy import spatial
from sklearn.model_selection import train_test_split, KFold

import entity.SymptomsDiagnosis
from utils.Constants import *

import util_cy

from pathlib import Path

################################################################################################################
#READ DATASET
################################################################################################################
os.chdir(CH_DIR) # to change current working dir
file_name = os.getcwd() + "/Symptoms-Diagnosis.txt"
f = open(file_name, "r").readlines()
orig_stdout = sys.stdout

admissions = dict()
for line in f:
    line.replace("\n", "")
    attributes = line.split(';')
    a = entity.SymptomsDiagnosis.SymptomsDiagnosis(attributes[entity.SymptomsDiagnosis.SymptomsDiagnosis.CONST_HADM_ID], attributes[entity.SymptomsDiagnosis.SymptomsDiagnosis.CONST_SUBJECT_ID], attributes[entity.SymptomsDiagnosis.SymptomsDiagnosis.CONST_ADMITTIME],
                                                   attributes[entity.SymptomsDiagnosis.SymptomsDiagnosis.CONST_DISCHTIME], attributes[entity.SymptomsDiagnosis.SymptomsDiagnosis.CONST_SYMPTOMS],
                                                   util_cy.preprocess_diagnosis(attributes[entity.SymptomsDiagnosis.SymptomsDiagnosis.CONST_DIAGNOSIS]))
    admissions.update({attributes[entity.SymptomsDiagnosis.SymptomsDiagnosis.CONST_HADM_ID]:a})

################################################################################################################
#LOAD MODEL
################################################################################################################
model = util_cy.load_model()

################################################################################################################
#COMPUTE EMBENDINGS
################################################################################################################
embendings_symptoms = util_cy.embending_symptoms(model,admissions)
embendings_diagnosis = util_cy.embending_diagnosis(model,admissions)

################################################################################################################
#PERFORMANCE MATRIX
################################################################################################################
performance_matrix_max = util_cy.init_performance_matrix()
performance_matrix_topK_max_dict = dict()
for topk in range(TOP_K_LOWER_BOUND, TOP_K_UPPER_BOUND, TOP_K_INCR):
    performance_matrix_topK_max_dict.update({topk: util_cy.init_performance_matrix()})

################################################################################################################
#OUTPUT DIRECTORIES
################################################################################################################
directory_prediction_root = os.getcwd() + '/Prediction Output_' + util_cy.current_time().replace('/','') + '/'
directory_prediction_details_root = os.getcwd() + '/Prediction Symptom Details_' + util_cy.current_time().replace('/','') + '/'
shutil.rmtree(directory_prediction_root, ignore_errors=True)
shutil.rmtree(directory_prediction_details_root, ignore_errors=True)
Path(directory_prediction_root).mkdir(parents=True, exist_ok=True)
Path(directory_prediction_details_root).mkdir(parents=True, exist_ok=True)
performance_out_file = open(directory_prediction_root + '/PerformanceIndex.txt', 'w')

################################################################################################################
# WORK ON SINGLE FOLD
################################################################################################################
for nFold in range(0,K_FOLD):
    directory_prediction = directory_prediction_root + 'Fold' + str(nFold) + "/"
    directory_prediction_details = directory_prediction_details_root + 'Fold' + str(nFold) + "/"
    shutil.rmtree(directory_prediction, ignore_errors=True)
    shutil.rmtree(directory_prediction_details, ignore_errors=True)
    Path(directory_prediction).mkdir(parents=True, exist_ok=True)
    Path(directory_prediction_details).mkdir(parents=True, exist_ok=True)
    ##########################################################################################################
    # LOAD TRAIN AND TEST SET
    ##########################################################################################################
    x_test = util_cy.load_dataset(nFold,TEST)
    x_train = util_cy.load_dataset(nFold,TRAIN)
    performance_out_file.write('\n FOLD %s: LEN train: %s, LEN test: %s \n' % (nFold, len(x_train), len(x_test)))
    print('FOLD %s: LEN train: %s, LEN test: %s' % (nFold, len(x_train), len(x_test)))
    ##########################################################################################################
    # COMPUTE PREDICTION
    ##########################################################################################################
    nrow = len(x_test)
    ncol = len(x_train)
    similarity_matrix = numpy.zeros(shape=(nrow, ncol))
    confusion_matrix_max = util_cy.init_confusion_matrix()
    confusion_matrix_Top_K_max_dict = dict()
    for topk in range(TOP_K_LOWER_BOUND, TOP_K_UPPER_BOUND, TOP_K_INCR):
        confusion_matrix_Top_K_max_dict.update({topk:util_cy.init_confusion_matrix()})
    # WORK ON SINGLE PREDICTION
    for i in range(0, nrow):
        # TEST SYMPTHOMS
        index = list(x_test[i].keys())[0]
        test_symptoms = list(x_test[i].values())[0]
        # TEST ADMISSION
        test_admission = admissions.get(index)
        #######################################################
        # EXECUTE PREDICTION
        #######################################################
        util_cy.predictS2V(i, index, test_admission, test_symptoms, x_train, nrow, ncol, embendings_symptoms, embendings_diagnosis,
                admissions, similarity_matrix, None, confusion_matrix_max, None, confusion_matrix_Top_K_max_dict,
                directory_prediction, directory_prediction_details, performance_out_file)
    ##########################################################################################################
    # COMPUTE PERFORMANCE INDEX
    ##########################################################################################################
    performance_out_file.write("PERFORMANCE INDEX of MAX SIMILARITY by MAX" + "\n")
    util_cy.compute_aggregated_performance_index(confusion_matrix_max, performance_matrix_max,nrow,performance_out_file)
    for topk in range(TOP_K_LOWER_BOUND, TOP_K_UPPER_BOUND, TOP_K_INCR):
        confusion_matrix_Top_K_max = confusion_matrix_Top_K_max_dict.get(topk)
        performance_matrix_topK_max = performance_matrix_topK_max_dict.get(topk)
        performance_out_file.write("\n PERFORMANCE INDEX of TOP-" + str(topk) + " SIMILARITY by MAX" + "\n")
        util_cy.compute_aggregated_performance_index(confusion_matrix_Top_K_max, performance_matrix_topK_max,nrow,performance_out_file)
#END BLOCK FOLD

##########################################################################################################
# COMPUTE MEAN PERFORMANCE INDEX FOR ALL FOLDS
##########################################################################################################
performance_out_file.write("************************************************************************************************************" + "\n")
performance_out_file.write("\n" + str(K_FOLD) + "-FOLD PERFORMANCE INDEX of MAX SIMILARITY by MAX" + "\n")
util_cy.print_performance_index(performance_matrix_max, performance_out_file)
performance_out_file.write("************************************************************************************************************" + "\n")
for topk in range(TOP_K_LOWER_BOUND, TOP_K_UPPER_BOUND, TOP_K_INCR):
    performance_matrix_topK_max = performance_matrix_topK_max_dict.get(topk)
    performance_out_file.write("\n" + str(K_FOLD) + "-FOLD PERFORMANCE INDEX of TOP-" + str(topk) + " SIMILARITY by MAX" + "\n")
    util_cy.print_performance_index(performance_matrix_topK_max, performance_out_file)
performance_out_file.write("************************************************************************************************************" + "\n")
performance_out_file.close()