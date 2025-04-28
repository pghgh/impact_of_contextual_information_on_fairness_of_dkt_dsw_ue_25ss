import logging

import torch
import lightning as L
import torch.nn as nn
from torch.optim import Adam

"""
TAKEN FROM 3

I adapted the code for implementing and training a neural network with a LSTM layer using PyTorch and Lightning
and for logging results in order to obtain plots with the help of TensorBoard from the following tutorial:

J. Starmer, “Long Short-Term Memory with PyTorch + Lightning”, StatQuest YouTube Channel, 24 Jan. 2023.
Available: https://www.youtube.com/watch?v=RHGiXPuo_pI
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
TAKEN FROM 10

I used the following resources for training and testing a model:

"Validate and test a model (basic)", Lightning AI https://lightning.ai/, Available: https://lightning.ai/docs/pytorch/stable/common/evaluation_basic.html

"LightningModule", Lightning AI https://lightning.ai/, Available: https://lightning.ai/docs/pytorch/stable/common/lightning_module.html

P. Lippe, "Tutorial 5: Inception, ResNet and DenseNet", GitHub Repository https://github.com/phlippe/uvadlc_notebooks,
16 Jun. 2023. Available: https://github.com/phlippe/uvadlc_notebooks/blob/master/docs/tutorial_notebooks/tutorial5/Inception_ResNet_DenseNet.ipynb
"""

"""
TAKEN FROM 13

My supervisor suggested using the Adam optimizer for my DKT implementation (in May 2023).

Furthermore, the choice of the learning rate was inspired by the value that the authors of the
following research papers used:

D. P. Kingma and J. L. Ba, “Adam: A Method for Stochastic Optimization,” arXiv
preprint arXiv: 1412.6980, 2017.

T. Gervet, K. Koedinger, J. Schneider, and T. Mitchell, “When is Deep Learning
the Best Approach to Knowledge Tracing?”, Journal of Educational Data Mining,
vol. 12, no. 3, pp. 31–54, 2020.
"""

"""
TAKEN FROM 18

With the help of my supervisor (in July 2023) I found out about the existence of computation graphs in PyTorch
and he suggested using the function "torch.stack" in order to keep it intact. I read about this topic
from the following source:

“Numerical optimization with pytorch.” Lecture notes of "Probabilistic Machine
Learning" course, University of Cambridge, https://www.cl.cam.ac.uk/teaching/2021/LE49/materials.html, 2020-2021.
Available: https://www.cl.cam.ac.uk/teaching/2021/LE49/probnn/3-3.pdf.
"""


# TAKEN FROM START 3
class DeepKnowledgeTracer(L.LightningModule):
    def __init__(self, input_data_info, learning_rate):
        super().__init__()
        self.input_data_info = input_data_info
        self.number_of_exercises = self.input_data_info.number_of_exercises
        self.input_vector_length = self.input_data_info.input_vector_length
        # TAKEN FROM START 6
        self.lstm = nn.LSTM(input_size=self.input_vector_length, hidden_size=self.number_of_exercises)
        self.dropout = nn.Dropout()
        self.linear = nn.Linear(in_features=self.number_of_exercises, out_features=self.number_of_exercises)
        self.sigmoid = nn.Sigmoid()
        # TAKEN FROM END 6
        self.learning_rate = learning_rate
        self.training_losses = []
        self.test_losses = []
        logging.info("The neural network has been created")

    def forward(self, input_data):
        # TAKEN FROM START 6
        lstm_layer_output = self.lstm(input_data)[0]
        dropout_output = self.dropout(lstm_layer_output)
        linear_layer_output = self.linear(dropout_output)
        neural_network_output = self.sigmoid(linear_layer_output)
        # TAKEN FROM END 6
        return neural_network_output

    def configure_optimizers(self):
        # TAKEN FROM START 13
        return Adam(self.parameters(), self.learning_rate)
        # TAKEN FROM END 13

    # TAKEN FROM START 10
    def training_step(self, batch, batch_idx):
        input_data, questions, answer_correctness_labels, student_id = batch
        neural_network_output = self.forward(input_data[0])
        # TAKEN FROM START 6
        loss_calculator = nn.BCELoss(reduction="sum")
        # TAKEN FROM END 6
        predictions_per_timestep = []
        timestep = 0
        for question in questions[0]:
            predictions_per_timestep.append(neural_network_output[timestep][int(question)])
            timestep += 1

        # TAKEN FROM START 18
        training_loss = loss_calculator(torch.stack(predictions_per_timestep), answer_correctness_labels[0])
        # TAKEN FROM END 18
        self.training_losses.append(training_loss)
        self.log(f"training_loss_student_no_{int(student_id)}", training_loss)
        timestep = 0
        for exercise_tag in questions[0]:
            self.log(f"training_step_prediction_student_no_{int(student_id)}_exercise_no_{exercise_tag}",
                     neural_network_output[timestep][int(int(exercise_tag))])
            timestep += 1
        return training_loss

    # TAKEN FROM END 10

    # TAKEN FROM START 10
    def test_step(self, batch, batch_idx):
        input_data, questions, answer_correctness_labels, student_id = batch
        neural_network_output = self.forward(input_data[0])
        # TAKEN FROM START 6
        loss_calculator = nn.BCELoss(reduction="sum")
        # TAKEN FROM END 6
        predictions_per_timestep = []
        timestep = 0
        for question in questions[0]:
            predictions_per_timestep.append(neural_network_output[timestep][int(question)])
            timestep += 1

        # TAKEN FROM START 18
        test_loss = loss_calculator(torch.stack(predictions_per_timestep), answer_correctness_labels[0])
        # TAKEN FROM END 18
        self.test_losses.append(test_loss)
        timestep = 0
        for exercise_tag in questions[0]:
            self.log(f"test_step_prediction_student_no_{int(student_id)}_exercise_no_{exercise_tag}",
                     neural_network_output[timestep][int(exercise_tag)])
            timestep += 1
        return test_loss
    # TAKEN FROM END 10


# TAKEN FROM END 3


class InputDataInfo:
    def __init__(self, number_of_exercises, input_vector_length):
        self.number_of_exercises = number_of_exercises
        self.input_vector_length = input_vector_length
