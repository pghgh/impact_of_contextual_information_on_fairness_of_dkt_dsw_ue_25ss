from experiments_dkt import *

def start_experiment(input_dataset, dataframe_synthetic_data):
    # if these parameters are modified, then a new round of experiments must be conducted
    seed = 1
    learned_skill_threshold = 0.5
    learning_rate = 0.001
    number_of_epochs = 10
    number_of_folds_cross_validation = 5
    number_of_students_receiving_recommendations = 10

    # these parameters can be modified within the same round of experiments
    input_data_mode = InputDataMode.CONTINUOUS

    # dataframe_synthetic_data is an empty dataframe if the chosen input_dataset contains real-world preprocessed data, and non-empty otherwise 
    experiments_dkt(input_dataset, dataframe_synthetic_data, input_data_mode, seed,
                    learned_skill_threshold,
                    learning_rate,
                    number_of_epochs,
                    number_of_folds_cross_validation,
                    number_of_students_receiving_recommendations)
