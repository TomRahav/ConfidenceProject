from typing import List, NamedTuple


class BatchResult(NamedTuple):
    """
    Represents the result of training for a single batch: the loss
    and number of correct classifications.
    """

    loss: float
    num_correct: int
    num_agree: int
    num_agree_wrong: int
    agreement_tail: float
    agreement_tail_distribution: float
    student_confidence: float


class EpochResult(NamedTuple):
    """
    Represents the result of training for a single epoch: the loss per batch
    and accuracy on the dataset (train or test).
    """

    losses: List[float]
    accuracy: float
    agreement: float
    agreement_wrong: float
    agreement_tail_list: List[float]
    agreement_tail_list_distribution: List[float]
    student_confidence_list: List[float]

class FitResult(NamedTuple):
    """
    Represents the result of fitting a model for multiple epochs given a
    training and test (or validation) set.
    The losses are for each batch and the accuracies are per epoch.
    """

    num_epochs: int
    train_loss: List[float]
    train_acc: List[float]
    train_agree: List[float]
    train_agree_wrong: List[float]
    train_tail_agreement: List[float]
    train_agreement_tail_distribution: List[float]
    test_loss: List[float]
    test_acc: List[float]
    test_agree: List[float]
    test_agree_wrong: List[float]
    test_tail_agreement: List[float]
    test_agreement_tail_distribution: List[float]
    student_confidence: List[float]