import logging
import csv
import random
import os

import numpy as np

"""
TAKEN FROM 1
I implemented this generator with the help of the explanations regarding synthetic data generation
and of a script example that my supervisor gave me (in March 2023).
"""

"""
TAKEN FROM 2
I chose the values for some parameters to be the same as the ones
used in the experiments presented in a research paper. The values were
given for two student groups, which were either fast or slow learners.

The parameters: number_of_students, p_L0, p_G, p_S, p_T (p_T_fast_learner or p_T_slow_learner)

The research paper: S. Tschiatschek, M. Knobelsdorf, and A. Singla, “Equity and Fairness of Bayesian
Knowledge Tracing,” in Proceedings of the 15th International Conference on Educational Data Mining
(A. Mitrovic and N. Bosch, eds.), (Durham, United Kingdom),
pp. 578–582, International Educational Data Mining Society, July 2022.

Note: the authors of the research paper mentioned above used, in turn, the same parameter values from the experiments
made by another group of researchers in the research paper mentioned below:

The research paper: S. Doroudi and E. Brunskill, “Fairer but Not Fair Enough On the Equitability of
Knowledge Tracing,” in Proceedings of the 9th International Conference on Learning
Analytics & Knowledge, LAK19, (New York, NY, USA), p. 335–339, Association for
Computing Machinery, 2019.

Note: I also made use of the parameters number_of_exercises and number_of_timesteps after my supervisor recommended me
some values for them. These parameters are also either mentioned or used in both research papers mentioned above.
"""

"""
TAKEN FROM 7
After reading an article about the usage of random number generators posted on the Matlab official documentation website
I decided to initialize the RNG only once in this program, with only one seed, because this approach is sufficient for
the majority of purposes. Although I did not use Matlab in this project, this information, in my opinion, can be applied
to RNGs offered by other sources (in my case, NumPy) as well.

7.1 "Controlling Random Number Generation", MathWorks https://www.mathworks.com/,
Available: https://www.mathworks.com/help/matlab/math/controlling-random-number-generation.html

I also read on the PyTorch official documentation website that for some cases one might need to manually set the seed of
Python.random as well.

7.2 "Reproducibility", PyTorch https://pytorch.org/, Available: https://pytorch.org/docs/stable/notes/randomness.html
"""

"""
TAKEN FROM 9

I took multiple configurations for the logger presented on this website:

V. Sajip, "Logging HOWTO", Python 3.11.5 documentation https://docs.python.org/3/,
Available: https://docs.python.org/3/howto/logging.html
"""


def generate_synthetic_data(synthetic_dataset, p_L0, p_S, p_G, p_T):
    logging.info("Generating synthetic data in file " + synthetic_dataset)
    with open(datasets_directory + synthetic_dataset, 'w+', newline='') as synthetic_dataset_csv:
        synthetic_dataset_writer = csv.writer(synthetic_dataset_csv, delimiter=',')
        # write the header line
        synthetic_dataset_writer.writerow(['StudentID', 'Timestep', 'ExerciseID', 'AnswerIsCorrect'])

    # TAKEN FROM START 2
    number_of_students = 200
    # we assume for simplicity that the number of skills equals the number of exercises,
    # and that each exercise covers exactly one skill
    number_of_exercises = 10
    number_of_timesteps = 100
    # TAKEN FROM END 2

    # TAKEN FROM START 1
    for student_id in range(0, number_of_students):

        exercises = np.zeros(number_of_exercises)
        for exercise_id in range(0, number_of_exercises):
            # we determine which exercises are already mastered
            exercises[exercise_id] = rng.binomial(n=1, p=p_L0, size=1)

        # beginning of the simulated interaction between the student and the ITS

        for timestep in range(0, number_of_timesteps):
            # an exercise to be solved is chosen
            exercise_id = rng.integers(low=0, high=number_of_exercises, size=1)[0]

            if exercises[exercise_id] == 0:  # if the student has not yet mastered the skill
                # we determine the answer correctness to the question
                # we note that the student might answer the question correctly by chance
                answer_correctness = rng.binomial(n=1, p=p_G, size=1)[0]
                # we determine whether the student achieves skill mastery this time
                exercises[exercise_id] = rng.binomial(n=1, p=p_T, size=1)
            else:  # the student has already mastered skill
                # we determine the answer correctness to the question
                # we note that the student might answer the question wrongly by mistake
                answer_correctness = rng.binomial(n=1, p=1 - p_S, size=1)[0]

            with open(datasets_directory + synthetic_dataset, 'a', newline='') as synthetic_dataset_csv:
                synthetic_dataset_writer = csv.writer(synthetic_dataset_csv, delimiter=',')
                synthetic_dataset_writer.writerow([student_id, timestep, exercise_id, answer_correctness])

    # TAKEN FROM END 1


def include_metadata(synthetic_dataset, metadata_types):
    synthetic_dataset_with_metadata = synthetic_dataset.split('.')[0] + "_with_metadata" + ".csv"
    logging.info("Generating metadata in file " + synthetic_dataset_with_metadata)
    with open(datasets_directory + synthetic_dataset, 'r', newline='') as synthetic_dataset_csv:
        synthetic_dataset_reader = csv.reader(synthetic_dataset_csv, delimiter=',')

        with open(datasets_directory + synthetic_dataset_with_metadata, 'w+',
                  newline='') as synthetic_dataset_with_metadata_csv:
            synthetic_dataset_with_metadata_writer = csv.writer(synthetic_dataset_with_metadata_csv, delimiter=',')
            synthetic_dataset_with_metadata_csv_header = ['StudentID', 'Timestep', 'ExerciseID', 'AnswerIsCorrect']
            for metadata in metadata_types:
                # append the metadata name to the header line
                synthetic_dataset_with_metadata_csv_header.append(metadata)
            synthetic_dataset_with_metadata_writer.writerow(synthetic_dataset_with_metadata_csv_header)

        next(synthetic_dataset_reader)  # skip the header line

        with open(datasets_directory + synthetic_dataset_with_metadata, 'a',
                  newline='') as synthetic_dataset_with_metadata_csv:
            synthetic_dataset_with_metadata_writer = csv.writer(synthetic_dataset_with_metadata_csv, delimiter=',')
            student_counter = -1.
            for existing_student_entry in synthetic_dataset_reader:
                next_student_number = float(existing_student_entry[0])
                new_student_entry = []
                for existing_element in existing_student_entry:
                    new_student_entry.append(existing_element)
                if next_student_number != student_counter:
                    metadata_elements = []
                    for metadata in metadata_types:
                        if metadata == "SocioeconomicStatus":
                            metadata_elements.append(generate_socioeconomic_status())
                        # elif other metadata types
                    student_counter = next_student_number
                new_student_entry += metadata_elements
                synthetic_dataset_with_metadata_writer.writerow(new_student_entry)


def generate_socioeconomic_status():
    # TAKEN FROM START 7.1
    number_of_possibilities = 2
    # we sample from the Uniform distribution, having only two possible values for this status, either 0 or 1
    socioeconomic_status = rng.integers(low=0, high=number_of_possibilities, size=1)[0]
    # TAKEN FROM END 7.1
    return socioeconomic_status


if __name__ == "__main__":
    # TAKEN FROM START 9
    logging.basicConfig(filename='synthetic_data_generator_logfile.log', format='%(asctime)s %(message)s',
                        level=logging.INFO)
    # TAKEN FROM END 9
    start_message = "Start of synthetic_data_generator.py script"
    print(start_message)
    logging.info("Start of synthetic_data_generator.py script")

    seed = 1
    # TAKEN FROM START 7.2
    random.seed(seed)
    # TAKEN FROM END 7.2
    rng = np.random.default_rng(seed)
    logging.info("seed value = %d", seed)

    # TAKEN FROM START 2
    # the BKT model parameters can be changed by the user
    p_L0 = 0
    p_S = 0.2
    p_G = 0.2
    p_T_fast_learner = 0.3
    p_T_slow_learner = 0.05
    # TAKEN FROM END 2
    logging.info("p_L0 = %f", p_L0)
    logging.info("p_S = %f", p_S)
    logging.info("p_G = %f", p_G)
    logging.info("p_T_fast_learner = %f", p_T_fast_learner)
    logging.info("p_T_slow_learner = %f", p_T_slow_learner)

    metadata_types = ["SocioeconomicStatus"]

    project_parent_directory = os.getcwd().split('src')[0]
    datasets_directory = project_parent_directory + "/src/res/datasets/"

    synthetic_datasets = ["synthetic_student_data_fast_learners.csv",
                          "synthetic_student_data_slow_learners.csv"]
    generate_synthetic_data(synthetic_datasets[0], p_L0, p_S, p_G, p_T_fast_learner)
    generate_synthetic_data(synthetic_datasets[1], p_L0, p_S, p_G, p_T_slow_learner)

    # add student metadata, if wished so
    if len(metadata_types) != 0:
        for synthetic_dataset in synthetic_datasets:
            include_metadata(synthetic_dataset, metadata_types)

    end_message = "End of synthetic_data_generator.py script"
    print(end_message)
    logging.info(end_message)
