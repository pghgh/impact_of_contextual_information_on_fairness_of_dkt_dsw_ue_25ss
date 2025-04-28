import logging
from enum import Enum, auto
from math import log, ceil
from datetime import datetime
import os

import torch
from torch.distributions import MultivariateNormal
from torch.utils.data import TensorDataset, DataLoader, ConcatDataset
from torch.nn.functional import one_hot
from torchmetrics.classification import BinaryAUROC, BinaryAccuracy

import lightning as L
import numpy as np
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from deep_knowledge_tracer import *

"""
TAKEN FROM 3

I adapted the code for implementing and training a neural network with a LSTM layer using PyTorch and Lightning
and for logging results in order to obtain plots with the help of TensorBoard from the following tutorial:

J. Starmer, “Long Short-Term Memory with PyTorch + Lightning”, StatQuest YouTube Channel, 24 Jan. 2023.
Available: https://www.youtube.com/watch?v=RHGiXPuo_pI
"""

"""
TAKEN FROM 4

I adapted the code for a plot example using Seaborn and Matplotlib and how to auto-detect if a suitable GPU is available
for performing the computations in PyTorch and Lightning or not from the following tutorial:

J. Starmer, “Introduction to Coding Neural Networks with PyTorch and Lightning”, StatQuest YouTube Channel,
19 Sep. 2022. Available: https://www.youtube.com/watch?v=khMzi6xPbuM

Note: the GPU of laptop on which I developed and tested the programs does not support CUDA,
so the CPU was used for performing all the computations.  
"""

"""
TAKEN FROM 5

I adapted the code for creating a Matplotlib figure with a table subplot and for adjusting
the position of a subplot from the following tutorials:

5.1 "Pyplot tutorial - Working with multiple figures and axes", Matplotlib https://matplotlib.org,
Available: https://matplotlib.org/stable/tutorials/introductory/pyplot.html

5.2 "Introduction to Figures", Matplotlib https://matplotlib.org,
Available: https://matplotlib.org/stable/users/explain/figure/figure_intro.html

5.3 "Table Demo", Matplotlib https://matplotlib.org,
Available: https://matplotlib.org/stable/gallery/misc/table_demo.html
"""

"""
TAKEN FROM 6
I made multiple design decisions for the neural network architecture and for the input data representation
based on the DKT implementation and experiments made by the authors of the following research paper:

C. Piech, J. Bassen, J. Huang, S. Ganguli, M. Sahami, L. Guibas, and J. Sohl-
Dickstein, “Deep Knowledge Tracing”, Home Page - Chris Piech, 2015. Available:
https://stanford.edu/~cpiech/bio/papers/deepKnowledgeTracing.pdf.
"""

"""
TAKEN FROM 8

I found out about the option of setting the same seed for the RNGs of Python.random, NumPy and PyTorch using only one 
PyTorch Lightning function when reading this tutorial:

P. Lippe, "Tutorial 5: Inception, ResNet and DenseNet", GitHub Repository https://github.com/phlippe/uvadlc_notebooks,
16 Jun. 2023. Available: https://github.com/phlippe/uvadlc_notebooks/blob/master/docs/tutorial_notebooks/tutorial5/Inception_ResNet_DenseNet.ipynb
"""

"""
TAKEN FROM 9

I took multiple configurations for the logger presented on this website:

V. Sajip, "Logging HOWTO", Python 3.11.5 documentation https://docs.python.org/3/,
Available: https://docs.python.org/3/howto/logging.html
"""

"""
TAKEN FROM 10

I used the following resources for training and testing a model:

"Validate and test a model (basic)", Lightning AI https://lightning.ai/, Available: https://lightning.ai/docs/pytorch/stable/common/evaluation_basic.html

"LightningModule", Lightning AI https://lightning.ai/, Available: https://lightning.ai/docs/pytorch/stable/common/lightning_module.html

P. Lippe, "Tutorial 5: Inception, ResNet and DenseNet", GitHub Repository https://github.com/phlippe/uvadlc_notebooks,
16 Jun. 2023. Available: https://github.com/phlippe/uvadlc_notebooks/blob/master/docs/tutorial_notebooks/tutorial5/Inception_ResNet_DenseNet.ipynb
"""

"""
TAKEN FROM 14

In order to be able to keep training the model after the .fit call ends (and also, after the epochs are finished),
I manually added some epochs. This solution was suggested in the following post:

C. Mocholí - "carmocca" GitHub user, issue opened on GitHub https://github.com/Lightning-AI/lightning/issues/11425 on 11 Jan. 2022,
answer given by the author of the solution on 3 Feb. 2022
"""

"""
TAKEN FROM 15

I found the code for renaming categorical data on this website:

"Categorical data", Pandas https://pandas.pydata.org/, Available: https://pandas.pydata.org/docs/user_guide/categorical.html
"""

"""
TAKEN FROM 16

I concatenated multiple datasets for the data loader by following an example shown on this website:

"Multiple datasets", PyTorch Lightning https://pytorch-lightning.readthedocs.io, Available: https://pytorch-lightning.readthedocs.io/en/1.0.8/multiple_loaders.html
"""

"""
TAKEN FROM 17

I used the following resources for implementing the 5-fold cross validation:

"Cross-validation: evaluating estimator performance", Scikit-learn https://scikit-learn.org/stable/, Available: https://scikit-learn.org/stable/modules/cross_validation.html

"sklearn.model_selection.StratifiedKFold", Scikit-learn https://scikit-learn.org/stable/, Available: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html
"""


def compute_most_probable_answer_correctness_label(question_tags, answer_correctness_labels, number_of_exercises, learned_skill_threshold):
    most_probable_answer_correctness_for_exercises = [-1. for _ in range(0, number_of_exercises)]
    for question_tag in question_tags:
        if most_probable_answer_correctness_for_exercises[int(question_tag)] == -1:
            counter_correct_answers = 0
            counter_wrong_answers = 0
            for index in range(0, len(question_tags)):
                if question_tag == question_tags[index]:
                    if answer_correctness_labels[index] == 1:
                        counter_correct_answers += 1
                    else:
                        counter_wrong_answers += 1
            if counter_correct_answers / (counter_correct_answers + counter_wrong_answers) >= learned_skill_threshold:
                # if for the majority of times an exercise was correctly solved by a student,
                # then we assume that the student gained that skill
                most_probable_answer_correctness_for_exercises[int(question_tag)] = 1.
            else:
                most_probable_answer_correctness_for_exercises[int(question_tag)] = 0.

    # for the cases in which some questions were not attempted by a student at all
    for _ in range(0, len(most_probable_answer_correctness_for_exercises)):
        if most_probable_answer_correctness_for_exercises[_] == -1.:
            most_probable_answer_correctness_for_exercises[_] = 0.

    return most_probable_answer_correctness_for_exercises


class InputDataMode(Enum):
    CONTINUOUS = auto()
    ONE_HOT_ENCODED = auto()
    COMPRESSED = auto()


def load_smaller_dataset(input_dataset):
    number_of_rows_to_load = 500
    dataset = pd.read_csv(input_dataset).head(number_of_rows_to_load)
    # Make the "QuestionId" data categorical
    dataset["QuestionId"] = dataset["QuestionId"].astype("category")
    # Assign to the questions an unique integer
    converted_question_ids = [unique_question_id for unique_question_id in range(0, dataset["QuestionId"].nunique())]
    # TAKEN FROM START 15
    dataset["QuestionId"] = dataset["QuestionId"].cat.rename_categories(converted_question_ids)
    # TAKEN FROM END 15
    return dataset


def experiments_dkt(input_dataset, dataframe_synthetic_data, input_data_mode, seed,
                    learned_skill_threshold,
                    learning_rate,
                    number_of_epochs,
                    number_of_folds_cross_validation,
                    number_of_students_receiving_recommendations):
    # TAKEN FROM START 9
    logging.basicConfig(filename='experiments_dkt_logfile.log', format='%(asctime)s %(message)s', level=logging.INFO)
    # TAKEN FROM END 9
    start_message = "Start of experiments_dkt.py script"
    print(start_message)
    logging.info(start_message)

    # TAKEN FROM START 8
    L.seed_everything(seed, workers=True)
    # TAKEN FROM END 8
    logging.info("seed value = %d", seed)

    # prepare the dataset from which the data will be read
    project_parent_directory = os.getcwd().split('experiments')[0]
    datasets_directory = project_parent_directory + "res/datasets/"
    plots_directory = project_parent_directory + "res/plots/"
    logging.info("Used input dataset: " + input_dataset)
    real_world_preprocessed_datasets = ["train_task_1_2_modified.csv",
                                        "train_task_1_2_modified_with_metadata.csv"]

    if input_dataset in real_world_preprocessed_datasets:
        dataset = load_smaller_dataset(datasets_directory + input_dataset)
    else:
        dataset = dataframe_synthetic_data
    dataset_column_names = dataset.columns

    # we subtract 2 because we don't count the first two columns, namely the student id and the timestep
    number_of_features = dataset_column_names.size - 2
    number_of_students = dataset[dataset_column_names[0]].nunique()
    number_of_exercises = dataset[dataset_column_names[2]].nunique()

    logging.info("number_of_students = %d", number_of_students)
    logging.info("number_of_exercises = %d", number_of_exercises)
    logging.info("number_of_features = %d", number_of_features)
    logging.info("input_data_mode = %s", input_data_mode.name.lower())
    logging.info("learned_skill_threshold = %f", learned_skill_threshold)
    logging.info("learning_rate = %f", learning_rate)
    logging.info("number_of_epochs = %d", number_of_epochs)
    logging.info("number_of_folds_cross_validation = %d", number_of_folds_cross_validation)
    logging.info("number_of_students_receiving_recommendations = %d", number_of_students_receiving_recommendations)
    # prepare data structures for each separate student
    question_tags = []
    answer_correctness_labels = []
    input_vectors = []
    standard_input = []  # vector containing standard DKT input pairs, namely question tag and answer correctness
    student_ids_with_frequencies = []
    frequencies_of_student_ids_with_frequencies = []
    # prepare data structures for evaluation metrics
    next_step_accuracies = []
    next_step_accuracies_random_method = []
    aucs = []
    aucs_random_method = []
    predicted_percentages_of_learned_skills = []
    training_losses = []
    test_losses = []
    training_losses_sizes = []
    test_losses_sizes = []

    # TAKEN FROM START 6
    # we prepare the elements necessary for creating compressed input data, in the case in which the user has chosen
    # this input data mode
    compressed_input_vector_length = ceil(log(2 ** (number_of_features - 1) * number_of_exercises))
    multivariate_normal_distribution = MultivariateNormal(torch.zeros(compressed_input_vector_length),
                                                          torch.eye(compressed_input_vector_length))
    # TAKEN FROM END 6
    input_vector_length = 0
    for dataset_info in dataset.iterrows():
        student_entry = dataset_info[1]

        student_id = float(student_entry[0])
        exercise_id = float(student_entry[2])
        answer_correctness = float(student_entry[3])
        socioeconomic_status = -1.
        if number_of_features > 2:
            socioeconomic_status = float(student_entry[4])

        question_tags.append(exercise_id)
        answer_correctness_labels.append(answer_correctness)
        student_ids_with_frequencies.append(student_id)

        if input_data_mode == InputDataMode.COMPRESSED:
            # TAKEN FROM START 6
            input_vector_length = compressed_input_vector_length
            random_vector_already_assigned = False
            random_vector_as_list = []
            # if a random vector has already been assigned to a certain standard input pair in the past, then we will
            # use the same random vector for the future occurrences of that pair
            if standard_input.count([exercise_id, answer_correctness]):
                existing_random_vector_index = standard_input.index([exercise_id, answer_correctness])
                random_vector_as_list = input_vectors[existing_random_vector_index]
                random_vector_already_assigned = True
            if random_vector_already_assigned is False:
                random_vector = multivariate_normal_distribution.sample()
                random_vector_as_list = [element.item() for element in random_vector]
            input_vectors.append(random_vector_as_list)
            # we keep track of the standard input pairs that have already been assigned a random vector
            standard_input.append([exercise_id, answer_correctness])
            # TAKEN FROM END 6
        elif input_data_mode == InputDataMode.ONE_HOT_ENCODED:
            # TAKEN FROM START 6
            input_vector_length = number_of_exercises * 2 ** (number_of_features - 1)
            input_vector_tag_number = exercise_id * 2 ** (number_of_features - 1) + answer_correctness
            if number_of_features > 2:
                input_vector_tag_number += socioeconomic_status
            one_hot_encoding = one_hot(torch.tensor(input_vector_tag_number).long(),
                                       num_classes=input_vector_length).float()
            input_vectors.append(one_hot_encoding.tolist())
            # TAKEN FROM END 6
        else:  # InputDataMode.CONTINUOUS
            input_vector_length = number_of_features
            standard_input_pair = [exercise_id, answer_correctness]
            if number_of_features > 2:
                standard_input_pair.append(socioeconomic_status)
            input_vectors.append(standard_input_pair)
    logging.info("input_vector_length = %d", input_vector_length)

    input_data_info = InputDataInfo(number_of_exercises, input_vector_length)
    knowledge_tracing_model = DeepKnowledgeTracer(input_data_info=input_data_info, learning_rate=learning_rate)

    frequency_counter = 0
    latest_student_id = student_ids_with_frequencies[0]
    for index in range(0, len(student_ids_with_frequencies) + 1):
        if index == len(student_ids_with_frequencies):
            if frequency_counter == 0:
                frequency_counter = 1
            for _ in range(0, frequency_counter):
                frequencies_of_student_ids_with_frequencies.append(frequency_counter)
        elif student_ids_with_frequencies[index] != latest_student_id:
            for _ in range(0, frequency_counter):
                frequencies_of_student_ids_with_frequencies.append(frequency_counter)
            frequency_counter = 1
            latest_student_id = student_ids_with_frequencies[index]
        else:
            frequency_counter += 1

    # TAKEN FROM START 3
    # we start training the neural network
    # TAKEN FROM START 4
    model_trainer = L.Trainer(max_epochs=number_of_epochs, accelerator="auto", devices="auto",
                              deterministic=True, enable_checkpointing=True)
    # TAKEN FROM END 4
    # TAKEN FROM START 17
    stratified_k_fold_cv = StratifiedKFold(n_splits=number_of_folds_cross_validation)
    data_splitter = stratified_k_fold_cv.split(student_ids_with_frequencies,
                                               frequencies_of_student_ids_with_frequencies)
    for fold_number in range(0, number_of_folds_cross_validation):
        logging.info("cross validation fold number = " + str(fold_number + 1))
        training_data_indices, test_data_indices = next(data_splitter)
        list_datasets = []
        list_input_vectors_per_student = []
        list_answer_correctness_labels_per_student = []
        question_tags_per_student = []

        latest_student_id = student_ids_with_frequencies[training_data_indices[0]]
        for index in range(0, len(training_data_indices) + 1):
            if index == len(training_data_indices):
                latest_student_id = student_ids_with_frequencies[training_data_indices[-1]]
                training_dataset = TensorDataset(
                    torch.tensor([list_input_vectors_per_student]),
                    torch.tensor([question_tags_per_student]),
                    torch.tensor([list_answer_correctness_labels_per_student]),
                    torch.tensor([latest_student_id]))
                list_datasets.append(training_dataset)
            elif student_ids_with_frequencies[training_data_indices[index]] != latest_student_id:
                training_dataset = TensorDataset(
                    torch.tensor([list_input_vectors_per_student]),
                    torch.tensor([question_tags_per_student]),
                    torch.tensor([list_answer_correctness_labels_per_student]),
                    torch.tensor([latest_student_id]))
                list_datasets.append(training_dataset)
                latest_student_id = student_ids_with_frequencies[training_data_indices[index]]
                list_input_vectors_per_student = []
                list_answer_correctness_labels_per_student = []
                question_tags_per_student = []
                list_input_vectors_per_student.append(input_vectors[training_data_indices[index]])
                list_answer_correctness_labels_per_student.append(answer_correctness_labels[training_data_indices[index]])
                question_tags_per_student.append(question_tags[training_data_indices[index]])
            else:
                list_input_vectors_per_student.append(input_vectors[training_data_indices[index]])
                list_answer_correctness_labels_per_student.append(answer_correctness_labels[training_data_indices[index]])
                question_tags_per_student.append(question_tags[training_data_indices[index]])

        # TAKEN FROM START 10
        # TAKEN FROM START 16
        training_dataloader = DataLoader(ConcatDataset(list_datasets))
        # TAKEN FROM END 16
        model_trainer.fit(knowledge_tracing_model, train_dataloaders=training_dataloader)
        # TAKEN FROM END 10

        training_losses_sizes.append(len(knowledge_tracing_model.training_losses))
        # TAKEN FROM START 14
        model_trainer.fit_loop.max_epochs += number_of_epochs
        # TAKEN FROM END 14
        list_datasets = []
        list_input_vectors_per_student = []
        list_answer_correctness_labels_per_student = []
        question_tags_per_student = []

        latest_student_id = student_ids_with_frequencies[test_data_indices[0]]
        for index in range(0, len(test_data_indices) + 1):
            if index == len(test_data_indices):
                latest_student_id = student_ids_with_frequencies[test_data_indices[-1]]
                test_dataset = TensorDataset(
                    torch.tensor([list_input_vectors_per_student]),
                    torch.tensor([question_tags_per_student]),
                    torch.tensor([list_answer_correctness_labels_per_student]),
                    torch.tensor([latest_student_id]))
                list_datasets.append(test_dataset)
            elif student_ids_with_frequencies[test_data_indices[index]] != latest_student_id:
                test_dataset = TensorDataset(
                    torch.tensor([list_input_vectors_per_student]),
                    torch.tensor([question_tags_per_student]),
                    torch.tensor([list_answer_correctness_labels_per_student]),
                    torch.tensor([latest_student_id]))
                # TAKEN FROM END 10
                list_datasets.append(test_dataset)
                latest_student_id = student_ids_with_frequencies[test_data_indices[index]]
                list_input_vectors_per_student = []
                list_answer_correctness_labels_per_student = []
                question_tags_per_student = []
                list_input_vectors_per_student.append(input_vectors[test_data_indices[index]])
                list_answer_correctness_labels_per_student.append(answer_correctness_labels[test_data_indices[index]])
                question_tags_per_student.append(question_tags[training_data_indices[index]])
            else:
                list_input_vectors_per_student.append(input_vectors[test_data_indices[index]])
                list_answer_correctness_labels_per_student.append(answer_correctness_labels[test_data_indices[index]])
                question_tags_per_student.append(question_tags[training_data_indices[index]])

        # TAKEN FROM START 10
        # TAKEN FROM START 16
        test_dataloader = DataLoader(ConcatDataset(list_datasets))
        # TAKEN FROM END 16
        model_trainer.test(knowledge_tracing_model, dataloaders=test_dataloader)
        # TAKEN FROM END 10
        test_losses_sizes.append(len(knowledge_tracing_model.test_losses))
    # TAKEN FROM END 17

    with open("exercise_recommendations.txt", 'a+', newline='') as exercise_recommendations_txt:
        exercise_recommendations_txt.write(
            str(datetime.now()) + " " + input_dataset + ", " + input_data_mode.name.lower() + " input\n")

    first_timestep_per_student_index = 0
    last_timestep_per_student_index = frequencies_of_student_ids_with_frequencies[first_timestep_per_student_index]
    for student_counter in range(0, number_of_students):
        dkt_prediction = torch.tensor(0.)
        if frequencies_of_student_ids_with_frequencies[first_timestep_per_student_index] == 1:
            dkt_prediction = knowledge_tracing_model(
                torch.tensor([input_vectors[first_timestep_per_student_index]])).detach()
        else:
            dkt_prediction = knowledge_tracing_model(
                torch.tensor(input_vectors[first_timestep_per_student_index:last_timestep_per_student_index])).detach()
        dkt_prediction = dkt_prediction[-1]

        # we now calculate what percentage of skills were learned by the students based on the predictions
        dkt_predictions_as_list = dkt_prediction.tolist()
        logging.info("Prediction for student no. %d is = " + ', '.join([str(_) for _ in dkt_predictions_as_list]),
                     student_ids_with_frequencies[first_timestep_per_student_index])
        number_of_learned_skills = 0
        for skill_prediction_probability in dkt_predictions_as_list:
            # if the probability of giving a correct answer to a question is at least a defined threshold,
            # then we assume that the student gained that skill (and will never forget it)
            if abs(skill_prediction_probability) >= learned_skill_threshold:
                number_of_learned_skills += 1
        predicted_percentages_of_learned_skills.append(number_of_learned_skills / number_of_exercises)
        # we compute evaluation metrics for each student, and then will average all the obtained results
        estimated_true_answer_correctness_labels = compute_most_probable_answer_correctness_label(
            question_tags[first_timestep_per_student_index:last_timestep_per_student_index],
            answer_correctness_labels[first_timestep_per_student_index:last_timestep_per_student_index], number_of_exercises,
            learned_skill_threshold)
        logging.info(
            "Estimated true labels for student no. %d are = " + ', '.join([str(_) for _ in estimated_true_answer_correctness_labels]),
            student_ids_with_frequencies[first_timestep_per_student_index])
        evaluation_metric = BinaryAccuracy()
        next_step_accuracy = evaluation_metric(preds=dkt_prediction, target=torch.tensor(estimated_true_answer_correctness_labels))
        next_step_accuracies.append(next_step_accuracy)
        evaluation_metric = BinaryAUROC()
        auc = evaluation_metric(preds=dkt_prediction, target=torch.tensor(estimated_true_answer_correctness_labels))
        aucs.append(auc)

        random_prediction = torch.tensor(np.random.uniform(low=0.0, high=1.0, size=number_of_exercises))
        evaluation_metric = BinaryAccuracy()
        next_step_accuracy = evaluation_metric(preds=random_prediction, target=torch.tensor(estimated_true_answer_correctness_labels))
        next_step_accuracies_random_method.append(next_step_accuracy)
        evaluation_metric = BinaryAUROC()
        auc = evaluation_metric(preds=random_prediction, target=torch.tensor(estimated_true_answer_correctness_labels))
        aucs_random_method.append(auc)

        if number_of_students_receiving_recommendations > 0:
            socioeconomic_statuses = [0., 1.]
            for index in range(0, number_of_features - 1):
                predictions_after_recommendations = []
                for exercise_index in range(0, number_of_exercises):
                    exercise_id = float(exercise_index)
                    answer_correctness = 1.
                    if input_data_mode == InputDataMode.COMPRESSED:
                        # TAKEN FROM START 6
                        random_vector_already_assigned = False
                        random_vector_as_list = []
                        if standard_input.count([exercise_id, answer_correctness]):
                            existing_random_vector_index = standard_input.index([exercise_id, answer_correctness])
                            random_vector_as_list = input_vectors[existing_random_vector_index]
                            random_vector_already_assigned = True
                        if random_vector_already_assigned is False:
                            random_vector = multivariate_normal_distribution.sample()
                            random_vector_as_list = [element.item() for element in random_vector]
                        new_input_datapoint = random_vector_as_list
                        # TAKEN FROM END 6
                    elif input_data_mode == InputDataMode.ONE_HOT_ENCODED:
                        # TAKEN FROM START 6
                        input_vector_length = number_of_exercises * 2 ** (number_of_features - 1)
                        input_vector_tag_number = exercise_id * 2 ** (number_of_features - 1) + answer_correctness
                        if number_of_features > 2:
                            input_vector_tag_number += socioeconomic_statuses[index]
                        one_hot_encoding = one_hot(torch.tensor(input_vector_tag_number).long(),
                                                   num_classes=input_vector_length).float()
                        new_input_datapoint = one_hot_encoding.tolist()
                        # TAKEN FROM END 6
                    else:  # InputDataMode.CONTINUOUS
                        standard_input_pair = [exercise_id, answer_correctness]
                        if number_of_features > 2:
                            standard_input_pair.append(socioeconomic_statuses[index])
                        new_input_datapoint = standard_input_pair

                    dkt_prediction = torch.tensor(0.)
                    if frequencies_of_student_ids_with_frequencies[first_timestep_per_student_index] == 1:
                        dkt_prediction = knowledge_tracing_model(torch.tensor(
                            [input_vectors[first_timestep_per_student_index]] + [new_input_datapoint])).detach()
                    else:
                        dkt_prediction = knowledge_tracing_model(torch.tensor(
                            input_vectors[first_timestep_per_student_index:last_timestep_per_student_index] + [
                                new_input_datapoint])).detach()
                    dkt_prediction = dkt_prediction[-1]
                    predictions_after_recommendations.append(sum(dkt_prediction.tolist()))

                student_who_received_recommendations = f"Recommended exercise for student no. " \
                                                       f"{int(student_ids_with_frequencies[first_timestep_per_student_index])}"
                recommended_exercise_message = f" is {predictions_after_recommendations.index(max(predictions_after_recommendations))}"
                with_metadata_message = ""
                if number_of_features > 2:
                    with_metadata_message = f" with socioeconomic status {socioeconomic_statuses[index]}"
                recommandation_message = student_who_received_recommendations + with_metadata_message + recommended_exercise_message
                print(recommandation_message)
                logging.info(recommandation_message)
                with open("exercise_recommendations.txt", 'a+', newline='') as exercise_recommendations_txt:
                    exercise_recommendations_txt.write(str(datetime.now()) + " " + recommandation_message + "\n")

        number_of_students_receiving_recommendations -= 1

        first_timestep_per_student_index = last_timestep_per_student_index
        if first_timestep_per_student_index < len(frequencies_of_student_ids_with_frequencies):
            last_timestep_per_student_index += frequencies_of_student_ids_with_frequencies[
                first_timestep_per_student_index]
    # TAKEN FROM END 3
    next_step_accuracies_as_list = [_.item() for _ in next_step_accuracies]
    aucs_as_list = [_.item() for _ in aucs]
    next_step_accuracy = sum(next_step_accuracies_as_list) / number_of_students
    auc = sum(aucs_as_list) / number_of_students

    next_step_accuracies_as_list_random_method = [_.item() for _ in next_step_accuracies_random_method]
    aucs_as_list_random_method = [_.item() for _ in aucs_random_method]
    next_step_accuracy_random_method = sum(next_step_accuracies_as_list_random_method) / number_of_students
    auc_random_method = sum(aucs_as_list_random_method) / number_of_students

    losses_start_size = 0
    losses_end_size = training_losses_sizes[0]
    for losses_sizes_index in range(1, len(training_losses_sizes) + 1):
        size_per_fold = int(
            len(knowledge_tracing_model.training_losses[losses_start_size:losses_end_size]) / number_of_epochs)
        start_size_per_epoch = losses_start_size
        end_size_per_epoch = losses_start_size + size_per_fold
        for _ in range(0, number_of_epochs):
            training_loss_per_epoch = sum(
                knowledge_tracing_model.training_losses[start_size_per_epoch:end_size_per_epoch]) / len(
                knowledge_tracing_model.training_losses[start_size_per_epoch:end_size_per_epoch])
            training_losses.append(training_loss_per_epoch)
            start_size_per_epoch += size_per_fold
            end_size_per_epoch += size_per_fold
        losses_start_size = losses_end_size
        if losses_sizes_index < len(training_losses_sizes):
            losses_end_size = training_losses_sizes[losses_sizes_index]

    losses_start_size = 0
    losses_end_size = test_losses_sizes[0]
    for losses_sizes_index in range(1, len(test_losses_sizes) + 1):
        size_per_fold = len(knowledge_tracing_model.test_losses[losses_start_size:losses_end_size])
        start_size_per_epoch = losses_start_size
        end_size_per_epoch = losses_start_size + size_per_fold
        test_loss_per_epoch = sum(knowledge_tracing_model.test_losses[start_size_per_epoch:end_size_per_epoch]) / len(
            knowledge_tracing_model.test_losses[start_size_per_epoch:end_size_per_epoch])
        test_losses.append(test_loss_per_epoch)
        losses_start_size = losses_end_size
        if losses_sizes_index < len(test_losses_sizes):
            losses_end_size = test_losses_sizes[losses_sizes_index]

    logging.info("next_step_accuracies = " + str(next_step_accuracies_as_list))
    logging.info("aucs = " + str(aucs_as_list))
    logging.info("next_step_accuracies_random_method = " + str(next_step_accuracies_as_list_random_method))
    logging.info("aucs_random_method = " + str(aucs_as_list_random_method))
    # TAKEN FROM START 5.1 5.2
    # plot the results
    plt.figure(1)
    # TAKEN FROM START 4
    sns.displot(x=predicted_percentages_of_learned_skills)
    plt.ylabel("frequency of occurrence")
    plt.xlabel("predicted % of learned skills")
    # TAKEN FROM END 4
    # TAKEN FROM START 5.3
    plt.table(rowLabels=["AUC", "Accuracy"],
              colLabels=[f"DKT prediction",
                         f"random method prediction"],
              cellText=[[str(auc), str(auc_random_method)],
                        [str(next_step_accuracy), str(next_step_accuracy_random_method)]],
              loc="top")
    plt.subplots_adjust(bottom=0.1, top=0.9, left=0.2)
    # TAKEN FROM END 5.3
    plt.savefig(plots_directory + "predictions_and_metrics_" + input_dataset.split('.')[
        0] + "_" + input_data_mode.name.lower() + "_input" + ".png")
    plt.clf()
    plt.close(1)
    # TAKEN FROM END 5.1 5.2

    training_losses = [_.item() for _ in training_losses]
    # TAKEN FROM START 5.1 5.2
    plt.figure(2)
    # TAKEN FROM START 4
    sns.lineplot(x=[_ for _ in range(1, len(training_losses) + 1)], y=training_losses)
    plt.ylabel("training loss")
    plt.xlabel("epoch number")
    # TAKEN FROM END 4
    # TAKEN FROM START 5.3
    plt.subplots_adjust(bottom=0.1, top=0.9, left=0.2)
    # TAKEN FROM END 5.3
    plt.savefig(plots_directory + "training_losses_" + input_dataset.split('.')[
        0] + "_" + input_data_mode.name.lower() + "_input" + ".png")
    plt.clf()
    plt.close(2)
    # TAKEN FROM END 5.1 5.2

    test_losses = [_.item() for _ in test_losses]
    # TAKEN FROM START 5.1 5.2
    plt.figure(3)
    # TAKEN FROM START 4
    sns.lineplot(x=[_ for _ in range(1, len(test_losses) + 1)], y=test_losses)
    plt.ylabel("test loss")
    plt.xlabel("epoch number")
    # TAKEN FROM END 4
    # TAKEN FROM START 5.3
    plt.subplots_adjust(bottom=0.1, top=0.9, left=0.2)
    # TAKEN FROM END 5.3
    plt.savefig(plots_directory + "test_losses_" + input_dataset.split('.')[
        0] + "_" + input_data_mode.name.lower() + "_input" + ".png")
    plt.clf()
    plt.close(3)
    # TAKEN FROM END 5.1 5.2

    with open(
            datasets_directory + "auc_and_accuracy_values_comparisons_" + input_data_mode.name.lower() + "_input" + ".csv",
            'a+', newline='') as evaluation_metrics_csv:
        try:
            evaluation_metrics_dataset = pd.read_csv(
                datasets_directory + "auc_and_accuracy_values_comparisons_" + input_data_mode.name.lower() + "_input" + ".csv")
        except:
            evaluation_metrics_csv.write("Dataset,AUC,Accuracy,ContainsMetadata\n")
        dataset_contains_metadata = False
        if "metadata" in input_dataset:
            dataset_contains_metadata = True
        dataset_abbreviation = "slow_learners"
        if "fast" in input_dataset:
            dataset_abbreviation = "fast_learners"
        if input_dataset in real_world_preprocessed_datasets:
            dataset_abbreviation = "real_learners"
        evaluation_metrics_csv.write(
            dataset_abbreviation + "," + str(auc) + "," + str(next_step_accuracy) + "," + str(
                dataset_contains_metadata) + "\n")

    evaluation_metrics_dataset = pd.read_csv(
        datasets_directory + "auc_and_accuracy_values_comparisons_" + input_data_mode.name.lower() + "_input" + ".csv")
    evaluation_metrics_dataset = evaluation_metrics_dataset.drop_duplicates()
    # TAKEN FROM START 5.1 5.2
    plt.figure(4)
    # TAKEN FROM START 4
    sns.barplot(data=evaluation_metrics_dataset, x="Dataset", y="AUC", hue="ContainsMetadata")
    plt.ylabel("AUC")
    plt.xlabel("datasets")
    # TAKEN FROM END 4
    # TAKEN FROM START 5.3
    plt.subplots_adjust(bottom=0.1, top=0.9, left=0.2)
    # TAKEN FROM END 5.3
    plt.savefig(plots_directory + "auc_comparisons_" + input_data_mode.name.lower() + "_input" + ".png")
    plt.clf()
    plt.close(4)
    # TAKEN FROM END 5.1 5.2

    # TAKEN FROM START 5.1 5.2
    plt.figure(5)
    # TAKEN FROM START 4
    sns.barplot(data=evaluation_metrics_dataset, x="Dataset", y="Accuracy", hue="ContainsMetadata")
    plt.ylabel("accuracy")
    plt.xlabel("datasets")
    # TAKEN FROM END 4
    # TAKEN FROM START 5.3
    plt.subplots_adjust(bottom=0.1, top=0.9, left=0.2)
    # TAKEN FROM END 5.3
    plt.savefig(plots_directory + "accuracy_comparisons_" + input_data_mode.name.lower() + "_input" + ".png")
    plt.clf()
    plt.close(5)
    # TAKEN FROM END 5.1 5.2

    end_message = "End of experiments_dkt.py script"
    print(end_message)
    logging.info(end_message)
