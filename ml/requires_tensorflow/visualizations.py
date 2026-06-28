from typing import Iterable

from numpy import floating
from numpy._typing import _64Bit

from libdpy.ml.data_types import LabeledData
from libdpy.ml.models import Model
from libdpy.ml.requires_tensorflow.early_stopping_utils import get_training_and_validation_loss
from libdpy.visualization.ml_plots import make_losses_per_epoch_figure


def get_losses(
    model_class, epochs_weights: Iterable, train_data: LabeledData, validation_data: LabeledData
) -> tuple[list[floating[_64Bit]], list[floating[_64Bit]]]:
    training_losses, validation_losses = [], []
    for epoch_weights in epochs_weights:
        model_of_wieghts = model_class([epoch_weights])
        training_loss, validation_loss = get_training_and_validation_loss(
            model_of_wieghts, train_data, validation_data
        )
        training_losses.append(training_loss)
        validation_losses.append(validation_loss)
    return training_losses, validation_losses


def make_losses_per_epoch_figure_from_model(
    model: Model, train_data: LabeledData, validation_data: LabeledData
):
    training_losses, validation_losses = get_losses(
        model.__class__, model.weights_arr, train_data, validation_data
    )
    return make_losses_per_epoch_figure(training_losses, validation_losses)
