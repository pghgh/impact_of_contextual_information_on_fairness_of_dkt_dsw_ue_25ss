import os
import logging

import pandas as pd

"""
TAKEN FROM 9

I took multiple configurations for the logger presented on this website:

V. Sajip, "Logging HOWTO", Python 3.11.5 documentation https://docs.python.org/3/,
Available: https://docs.python.org/3/howto/logging.html
"""

"""
TAKEN FROM 11

I also used the real-world data collected from the Eedi educational platform.
The dataset can be downloaded here: https://eedi.com/projects/neurips-education-challenge

For the experiments, I preprocessed the data, so after running this script, the user can obtain modified versions of the
files, with content which suits the purpose of this project. I didn't upload the modified datasets to the repository.

The license of the data is a "Attribution-NonCommercial-NoDerivatives 4.0 International (CC BY-NC-ND 4.0)" license:
https://creativecommons.org/licenses/by-nc-nd/4.0/.

The research paper from which I found out about the dataset is the following:

Z. Wang, A. Lamb, E. Saveliev, P. Cameron, Y. Zaykov, J. M. Hernández-Lobato,
R. E. Turner, R. G. Baraniuk, C. Barton, S. P. Jones, S. Woodhead, and C. Zhang,
“Results and Insights from Diagnostic Questions: The NeurIPS 2020 Education
Challenge,” arXiv preprint arXiv: 2104.04034, 2021.

The research paper which announced the tasks of the NeurIPS 2020 Education Challenge is the following:

Z. Wang, A. Lamb, E. Saveliev, P. Cameron, Y. Zaykov, J. M. Hernández-Lobato,
R. E. Turner, R. G. Baraniuk, C. Barton, S. P. Jones, S. Woodhead, and C. Zhang,
“Instructions and Guide for Diagnostic Questions: The NeurIPS 2020 Education
Challenge,” arXiv preprint arXiv:2007.12061, 2021.
"""

"""
TAKEN FROM 12

I found the code for performing a left-join operation using a certain column on this website:

"pandas.DataFrame.join", Pandas https://pandas.pydata.org/, 
Available: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.join.html
"""

"""
TAKEN FROM 15

I found the code for renaming categorical data on this website:

"Categorical data", Pandas https://pandas.pydata.org/, 
Available: https://pandas.pydata.org/docs/user_guide/categorical.html
"""


def generate_modified_dataset(initial_dataset_csv, modified_dataset_csv):
    logging.info("Start of generate_modified_dataset function")
    # so that we don't run out of memory while processing the data, we will load it in chunks;
    # this means that only a limited amount of lines will be loaded at a time
    chunk_size = 300000
    logging.info("chunk_size = " + str(chunk_size))
    chunk_index = 0

    modified_dataset = pd.DataFrame(columns=["AnswerId", "UserId", "DateAnswered", "QuestionId", "IsCorrect"])
    while chunk_size != 0:
        logging.info("chunk_index = " + str(chunk_index))
        relevant_columns = ["AnswerId", "UserId", "QuestionId", "IsCorrect"]
        modified_dataset_handler = pd.read_csv(datasets_directory + initial_dataset_csv, usecols=relevant_columns,
                                               skiprows=[_ for _ in range(1, chunk_index * chunk_size)]).head(
            chunk_size)
        # if the last chunk of data consists of fewer lines than chunk_size,
        # it means that there is no more data available for loading, so we will soon stop the loading process
        if modified_dataset_handler.shape[0] != chunk_size:
            chunk_size = 0
        relevant_columns = ["AnswerId", "DateAnswered"]
        answer_metadata_dataset_handler = pd.read_csv(datasets_directory + answer_metadata_dataset_csv,
                                                      usecols=relevant_columns,
                                                      skiprows=[_ for _ in range(1, chunk_index * chunk_size)]).head(
            chunk_size)
        # TAKEN FROM START 12
        # Perform a left join on the common column "AnswerId" between the two datasets
        modified_dataset_handler = modified_dataset_handler.join(answer_metadata_dataset_handler.set_index('AnswerId'),
                                                                 on='AnswerId')
        # TAKEN FROM END 12
        # Remove rows if they do not have a valid "DateAnswered" entry, since DKT makes use of time series
        modified_dataset_handler = modified_dataset_handler[modified_dataset_handler['DateAnswered'].notna()]
        # Reorder the columns in a format already established for the input datasets of DKT
        modified_dataset_handler = pd.DataFrame(data=modified_dataset_handler,
                                                columns=["UserId", "DateAnswered", "QuestionId", "IsCorrect"])
        if chunk_index == 0:
            # if this is the first chunk, we just copy the contents to modified_dataset
            modified_dataset = modified_dataset_handler.copy()
        else:
            # otherwise we merge the already existing contents with the new ones
            modified_dataset = pd.merge_ordered(modified_dataset, modified_dataset_handler)
        chunk_index += 1

    modified_dataset = modified_dataset.sort_values(by=['UserId', 'DateAnswered'])

    # Make the "DateAnswered" data categorical
    modified_dataset["DateAnswered"] = modified_dataset["DateAnswered"].astype("category")
    # Assign to the timesteps an unique integer
    converted_timesteps = [unique_timestep for unique_timestep in range(0, modified_dataset["DateAnswered"].nunique())]
    # TAKEN FROM START 15
    modified_dataset["DateAnswered"] = modified_dataset["DateAnswered"].cat.rename_categories(converted_timesteps)
    # TAKEN FROM END 15

    # Make the "QuestionId" data categorical
    modified_dataset["QuestionId"] = modified_dataset["QuestionId"].astype("category")
    # Assign to the questions an unique integer
    converted_question_ids = [unique_question_id for unique_question_id in
                              range(0, modified_dataset["QuestionId"].nunique())]
    # TAKEN FROM START 15
    modified_dataset["QuestionId"] = modified_dataset["QuestionId"].cat.rename_categories(converted_question_ids)
    # TAKEN FROM END 15

    # drop duplicates (if there are any, just to be sure)
    modified_dataset.drop_duplicates()
    logging.info("Writing the modified dataset in file " + modified_dataset_csv)
    modified_dataset.to_csv(path_or_buf=datasets_directory + modified_dataset_csv, index=False)
    info_about_dataset_size = modified_dataset_csv + f"\nno. of students = {modified_dataset['UserId'].nunique()} \n " \
                                                     f"no. of questions = {modified_dataset['QuestionId'].nunique()} " \
                                                     f"\nno. of answers = {modified_dataset.shape[0]}"
    logging.info(info_about_dataset_size)


def generate_modified_dataset_with_premium_pupil_metadata(initial_dataset_csv, modified_dataset_csv):
    logging.info("Start of generate_modified_dataset_with_metadata function")
    chunk_size = 300000
    logging.info("chunk_size = " + str(chunk_size))
    chunk_index = 0

    modified_dataset = pd.DataFrame(columns=["AnswerId", "UserId", "DateAnswered", "QuestionId", "IsCorrect"])
    while chunk_size != 0:
        logging.info("chunk_index = " + str(chunk_index))
        relevant_columns = ["UserId", "DateAnswered", "QuestionId", "IsCorrect"]
        modified_dataset = pd.read_csv(datasets_directory + initial_dataset_csv, usecols=relevant_columns)

        relevant_columns = ["UserId", "PremiumPupil"]
        student_metadata_dataset_handler = pd.read_csv(datasets_directory + student_metadata_dataset_csv,
                                                       usecols=relevant_columns,
                                                       skiprows=[_ for _ in range(1, chunk_index * chunk_size)]).head(
            chunk_size)
        if student_metadata_dataset_handler.shape[0] != chunk_size:
            chunk_size = 0
        # TAKEN FROM START 12
        # Perform a left join on the common column "UserId" between the two datasets
        modified_dataset = modified_dataset.join(student_metadata_dataset_handler.set_index('UserId'), on='UserId')
        # TAKEN FROM END 12
        modified_dataset = modified_dataset.sort_values(by=['UserId'])
        # Remove rows if they do not have a valid "PremiumPupil" entry, since we want to make use of the metadata
        modified_dataset = modified_dataset[modified_dataset['PremiumPupil'].notna()]
        # Reorder the columns in a format already established for the input datasets of DKT
        modified_dataset = pd.DataFrame(data=modified_dataset,
                                        columns=["UserId", "DateAnswered", "QuestionId", "IsCorrect", "PremiumPupil"])
        chunk_index += 1

    # drop duplicates (if there are any, just to be sure)
    modified_dataset.drop_duplicates()
    logging.info("Writing the modified dataset in file " + modified_dataset_csv)
    modified_dataset.to_csv(path_or_buf=datasets_directory + modified_dataset_csv, index=False)
    info_about_dataset_size = modified_dataset_csv + f"\nno. of students = {modified_dataset['UserId'].nunique()} " \
                                                     f"\nno. of questions = {modified_dataset['QuestionId'].nunique()} " \
                                                     f"\nno. of answers = {modified_dataset.shape[0]}"
    logging.info(info_about_dataset_size)


if __name__ == "__main__":
    # TAKEN FROM START 9
    logging.basicConfig(filename='eedi_platform_data_preparator_logfile.log', format='%(asctime)s %(message)s',
                        level=logging.INFO)
    # TAKEN FROM END 9
    start_message = "Start of eedi_platform_data_preparator.py script"
    print(start_message)
    logging.info("Start of eedi_platform_data_preparator.py script")

    project_parent_directory = os.getcwd().split('src')[0]
    datasets_directory = project_parent_directory + "src/res/datasets/"

    # TAKEN FROM START 11
    training_dataset_csv = "train_task_1_2.csv"
    student_metadata_dataset_csv = "student_metadata_task_1_2.csv"
    answer_metadata_dataset_csv = "answer_metadata_task_1_2.csv"
    # TAKEN FROM END 11
    training_dataset_modified_csv = "train_task_1_2_modified.csv"
    training_output_modified_with_metadata_csv = "train_task_1_2_modified_with_metadata.csv"

    generate_modified_dataset(training_dataset_csv, training_dataset_modified_csv)
    generate_modified_dataset_with_premium_pupil_metadata(training_dataset_modified_csv,
                                                          training_output_modified_with_metadata_csv)

    end_message = "End of eedi_platform_data_preparator.py script"
    print(end_message)
    logging.info("End of eedi_platform_data_preparator.py script")
