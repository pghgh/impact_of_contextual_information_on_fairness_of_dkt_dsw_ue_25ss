# The Impact of Contextual Information on the Fairness of Deep Knowledge Tracing

## Technical details

- Python version: 3.12
- The usage of a virtual environment is recommended - "venv" was used for this project. The file "requirements.txt" contains the project dependencies and their versions.
- In the Jupyter Notebook file "handle\_dbrepo\_data.ipynb", training and test sets of synthetic data which was uploaded on the test instance of DBRepo is imported. Afterwards, the code from multiple Python files is executed. Only the training set from DBRepo is used for the training and test sets of the model. 

## Background information

The files inside the "src" folder were copy-pasted from the initial project repository, with only minor changes in the code so that it is compatible with the code from the Jupyter Notebook file. The initial project is my Bachelor's thesis project, completed under the supervision of Ass.-Prof. Dipl.-Ing.
Dr.techn. Sebastian Tschiatschek, BSc, from the University of Vienna. 

The file upload took place in the context of the assignment of the [194.045 Data Stewardship UE 2025SS](https://tiss.tuwien.ac.at/course/courseDetails.xhtml?dswid=9771&dsrid=330&semester=2025S&courseNr=194045) course from the Technical University of Vienna. Because of the license of the real-world datasets that I downloaded from the respective [Eedi project site](https://eedi.com/projects/neurips-education-challenge) also mentioned in "eedi_platform_data_preparator.py", the .csv files being "train_task_1_2.csv", "student_metadata_task_1_2.csv" and "answer_metadata_task_1_2.csv", I chose to use only the datasets that I synthetically generated and which were published under another license for this assignment.
