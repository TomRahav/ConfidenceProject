import os
import abc
import sys
import torch
import torch.nn as nn
import tqdm.auto
import numpy as np
from typing import Any, Tuple, Callable, Optional, cast
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.nn import KLDivLoss, LogSoftmax, Softmax


from train_results import FitResult, BatchResult, EpochResult

from classifier import Classifier

class Trainer(abc.ABC):
    """
    A class abstracting the various tasks of training models.

    Provides methods at multiple levels of granularity:
    - Multiple epochs (fit)
    - Single epoch (train_epoch/test_epoch)
    - Single batch (train_batch/test_batch)
    """

    def __init__(
        self,
        model: nn.Module,
        device: Optional[torch.device] = None,

    ):
        """
        Initialize the trainer.
        :param model: Instance of the model to train.
        :param device: torch.device to run training on (CPU or GPU).
        """
        self.model = model
        self.device = device

        if self.device:
            model.to(self.device)

    def fit(
        self,
        dl_train: DataLoader,
        dl_test: DataLoader,
        num_epochs: int,
        run = None,
        checkpoints: str = None,
        early_stopping: int = None,
        print_every: int = 1,
        bins=10,
        **kw,
    ) -> FitResult:
        """
        Trains the model for multiple epochs with a given training set,
        and calculates validation loss over a given validation set.
        :param dl_train: Dataloader for the training set.
        :param dl_test: Dataloader for the test set.
        :param num_epochs: Number of epochs to train for.
        :param checkpoints: Whether to save model to file every time the
            test set accuracy improves. Should be a string containing a
            filename without extension.
        :param early_stopping: Whether to stop training early if there is no
            test loss improvement for this number of epochs.
        :param print_every: Print progress every this number of epochs.
        :return: A FitResult object containing train and test losses per epoch.
        """
        actual_num_epochs = 0
        epochs_without_improvement = 0

        train_loss, train_acc, train_agree, train_agree_wrong, train_tail_agreement, train_tail_agreement_distribution, train_student_confidence, train_ece, train_mce = [], [], [], [], [], [], [], [], []
        test_loss, test_acc, test_agree, test_agree_wrong, test_tail_agreement, test_tail_agreement_distribution, test_student_confidence, test_ece, test_mce = [], [], [], [], [], [], [], [], []
        best_acc = None
        best_agr = None

        for epoch in range(num_epochs):
            verbose = False  # pass this to train/test_epoch.
            if print_every > 0 and (
                epoch % print_every == 0 or epoch == num_epochs - 1
            ):
                verbose = True
            self._print(f"--- EPOCH {epoch+1}/{num_epochs} ---", verbose)

            # TODO: Train & evaluate for one epoch
            #  - Use the train/test_epoch methods.
            #  - Save losses and accuracies in the lists above.
            # ====== YOUR CODE: ======
            actual_num_epochs += 1
            train_result = self.train_epoch(dl_train, verbose=verbose, **kw)
            train_loss.append(sum(train_result.losses)/len(train_result.losses))
            train_acc.append(train_result.accuracy)
            train_agree.append(train_result.agreement)
            train_agree_wrong.append(train_result.agreement_wrong)
            train_tail_agreement.append(sum(train_result.agreement_tail_list)/len(train_result.agreement_tail_list))
            train_tail_agreement_distribution.append(sum(train_result.agreement_tail_list_distribution)/len(train_result.agreement_tail_list_distribution))
            train_student_confidence.append(sum(train_result.student_confidence_list) / len(train_result.student_confidence_list))
            train_ece.append(np.sum((np.abs(train_result.acc_per_bin - train_result.conf_per_bin) * train_result.num_per_bin)) / np.sum(train_result.num_per_bin))
            train_mce.append(np.max(np.abs(train_result.acc_per_bin - train_result.conf_per_bin)))
            print("Train Epoch Level", train_result.num_per_bin, train_result.acc_per_bin, train_result.conf_per_bin)

            with torch.no_grad():
                test_result = self.test_epoch(dl_test, verbose=verbose, **kw)
            test_loss.append(sum(test_result.losses)/len(test_result.losses))
            test_acc.append(test_result.accuracy)
            test_agree.append(test_result.agreement)
            test_agree_wrong.append(test_result.agreement_wrong)
            test_tail_agreement.append(sum(test_result.agreement_tail_list)/len(test_result.agreement_tail_list))
            test_tail_agreement_distribution.append(sum(test_result.agreement_tail_list_distribution)/len(test_result.agreement_tail_list_distribution))
            test_student_confidence.append(sum(test_result.student_confidence_list) / len(test_result.student_confidence_list))
            test_ece.append(sum((abs(test_result.acc_per_bin - test_result.conf_per_bin) * test_result.num_per_bin)) / sum(test_result.num_per_bin))
            test_mce.append(max(abs(test_result.acc_per_bin - test_result.conf_per_bin)))
            print("Test Epoch Level", test_result.num_per_bin, test_result.acc_per_bin, test_result.conf_per_bin)

            if run != None:
                run.log({
                    "Train Loss": sum(train_result.losses)/len(train_result.losses),
                    "Train Accuracy": train_result.accuracy,
                    "Train Agreement": train_result.agreement, 
                    "Train Agreement Wrong": train_result.agreement_wrong,
                    "Train Tail Agreement Not Distribution": sum(train_result.agreement_tail_list)/len(train_result.agreement_tail_list),
                    "Train Tail Agreement Distribution": sum(train_result.agreement_tail_list_distribution)/len(train_result.agreement_tail_list_distribution),
                    "Train Student Confidence": sum(train_result.student_confidence_list)/len(train_result.student_confidence_list),
                    "Train ECE": sum((abs(train_result.acc_per_bin - train_result.conf_per_bin) * train_result.num_per_bin)) / sum(train_result.num_per_bin),
                    "Train MCE": max(abs(train_result.acc_per_bin - train_result.conf_per_bin)),
                    "Test Loss": sum(test_result.losses)/len(test_result.losses),
                    "Test Accuracy": test_result.accuracy,
                    "Test Agreement": test_result.agreement,
                    "Test Agreement Wrong": test_result.agreement_wrong,
                    "Test Tail Agreement Not Distribution": sum(test_result.agreement_tail_list)/len(test_result.agreement_tail_list),
                    "Test Tail Agreement Distribution": sum(test_result.agreement_tail_list_distribution)/len(test_result.agreement_tail_list_distribution),
                    "Test Student Confidence": sum(test_result.student_confidence_list)/len(test_result.student_confidence_list),
                    "Test ECE": sum((abs(test_result.acc_per_bin - test_result.conf_per_bin) * test_result.num_per_bin)) / sum(test_result.num_per_bin),
                    "Test MCE": max(abs(test_result.acc_per_bin - test_result.conf_per_bin))
                    })
            # ========================

            # TODO:
            #  - Optional: Implement early stopping. This is a very useful and
            #    simple regularization technique that is highly recommended.
            #  - Optional: Implement checkpoints. You can use the save_checkpoint
            #    method on this class to save the model to the file specified by
            #    the checkpoints argument.
            if best_agr is None or test_result.agreement > best_agr:
                # ====== YOUR CODE: ======
                epochs_without_improvement = 0
                best_agr = test_result.agreement
                if checkpoints is not None:
                    self.save_checkpoint(f"{checkpoints}_agr.pt")
                # ========================
            elif best_acc is None or test_result.accuracy > best_acc:
                # ====== YOUR CODE: ======
                epochs_without_improvement = 0
                best_acc = test_result.accuracy
                if checkpoints is not None:
                    self.save_checkpoint(f"{checkpoints}_acc.pt")
                # ========================
            else:
                # ====== YOUR CODE: ======
                if early_stopping is not None:
                    epochs_without_improvement += 1
                    if epochs_without_improvement == early_stopping:
                        return FitResult(actual_num_epochs, train_loss, train_acc, train_agree, train_agree_wrong, train_tail_agreement, train_tail_agreement_distribution, train_student_confidence, test_loss, test_acc, test_agree, test_agree_wrong, test_tail_agreement, test_tail_agreement_distribution, test_student_confidence)
                # ========================

        return FitResult(actual_num_epochs, train_loss, train_acc, train_agree, train_agree_wrong, train_tail_agreement, train_tail_agreement_distribution, train_student_confidence, test_loss, test_acc, test_agree, test_agree_wrong, test_tail_agreement, test_tail_agreement_distribution, test_student_confidence)

    def save_checkpoint(self, checkpoint_filename: str):
        """
        Saves the model in it's current state to a file with the given name (treated
        as a relative path).
        :param checkpoint_filename: File name or relative path to save to.
        """
        torch.save(self.model, checkpoint_filename)
        print(f"\n*** Saved checkpoint {checkpoint_filename}")

    def train_epoch(self, dl_train: DataLoader, **kw) -> EpochResult:
        """
        Train once over a training set (single epoch).
        :param dl_train: DataLoader for the training set.
        :param kw: Keyword args supported by _foreach_batch.
        :return: An EpochResult for the epoch.
        """
        self.model.train(True)  # set train mode
        return self._foreach_batch(dl_train, self.train_batch, **kw)

    def test_epoch(self, dl_test: DataLoader, **kw) -> EpochResult:
        """
        Evaluate model once over a test set (single epoch).
        :param dl_test: DataLoader for the test set.
        :param kw: Keyword args supported by _foreach_batch.
        :return: An EpochResult for the epoch.
        """
        self.model.train(False)  # set evaluation (test) mode
        return self._foreach_batch(dl_test, self.test_batch, **kw)

    @abc.abstractmethod
    def train_batch(self, batch) -> BatchResult:
        """
        Runs a single batch forward through the model, calculates loss,
        preforms back-propagation and updates weights.
        :param batch: A single batch of data  from a data loader (might
            be a tuple of data and labels or anything else depending on
            the underlying dataset.
        :return: A BatchResult containing the value of the loss function and
            the number of correctly classified samples in the batch.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def test_batch(self, batch) -> BatchResult:
        """
        Runs a single batch forward through the model and calculates loss.
        :param batch: A single batch of data  from a data loader (might
            be a tuple of data and labels or anything else depending on
            the underlying dataset.
        :return: A BatchResult containing the value of the loss function and
            the number of correctly classified samples in the batch.
        """
        raise NotImplementedError()

    @staticmethod
    def _print(message, verbose=True):
        """Simple wrapper around print to make it conditional"""
        if verbose:
            print(message)

    @staticmethod
    def _foreach_batch(
        dl: DataLoader,
        forward_fn: Callable[[Any], BatchResult],
        verbose=True,
        max_batches=None,
        bins=10,
    ) -> EpochResult:
        """
        Evaluates the given forward-function on batches from the given
        dataloader, and prints progress along the way.
        """
        losses = []
        num_correct = 0
        num_agree = 0
        num_agree_wrong = 0
        agreement_tail_list = []
        agreement_tail_list_distribution = []
        student_confidence_list = []
        num_samples = len(dl.sampler)
        num_batches = len(dl.batch_sampler)
        num_per_bin, acc_per_bin, conf_per_bin = np.zeros(bins, dtype=int), np.zeros(bins, dtype=int), np.zeros(bins, dtype=float)


        if max_batches is not None:
            if max_batches < num_batches:
                num_batches = max_batches
                num_samples = num_batches * dl.batch_size

        if verbose:
            pbar_fn = tqdm.auto.tqdm
            pbar_file = sys.stdout
        else:
            pbar_fn = tqdm.tqdm
            pbar_file = open(os.devnull, "w")

        pbar_name = forward_fn.__name__
        with pbar_fn(desc=pbar_name, total=num_batches, file=pbar_file) as pbar:
            dl_iter = iter(dl)
            for batch_idx in range(num_batches):
                data = next(dl_iter)
                batch_res = forward_fn(data)

                pbar.set_description(f"{pbar_name} ({batch_res.loss:.3f})")
                pbar.update()

                losses.append(batch_res.loss)
                num_correct += batch_res.num_correct
                num_agree += batch_res.num_agree
                num_agree_wrong += batch_res.num_agree_wrong
                agreement_tail_list.append(batch_res.agreement_tail)
                agreement_tail_list_distribution.append(batch_res.agreement_tail_distribution)
                student_confidence_list.append(batch_res.student_confidence)
                num_per_bin += batch_res.num_per_bin
                acc_per_bin += batch_res.acc_per_bin
                conf_per_bin += batch_res.conf_per_bin
            avg_loss = sum(losses) / num_batches
            accuracy = 100.0 * num_correct / num_samples
            agreement = 100.0 * num_agree / num_samples
            student_confidence = 100.0 * sum(student_confidence_list) / len(student_confidence_list)
            if num_correct == num_samples:
                agreement_wrong = 100.0
            else:
                agreement_wrong = 100.0 * num_agree_wrong / (num_samples - num_correct)

            pbar.set_description(
                f"{pbar_name} "
                f"(Avg. Loss {avg_loss:.3f}, "
                f"Accuracy {accuracy:.1f}, "
                f"Agreement {agreement:.1f}, "
                f"Agreement Wrong {agreement_wrong:.1f})"
                f"Student Confidence {student_confidence:.1f}"
            )

        if not verbose:
            pbar_file.close()

        return EpochResult(losses, accuracy, agreement, agreement_wrong, agreement_tail_list, agreement_tail_list_distribution, student_confidence_list, num_per_bin, acc_per_bin, conf_per_bin)

class DistillationTrainer(Trainer):
    """
    Trainer for our Classifier-based models.
    """

    def __init__(
        self,
        model: Classifier,
        teacher_model: Classifier,
        loss_fn: nn.Module,
        optimizer: Optimizer,
        device: Optional[torch.device] = None,
        teacher_temp = 1,
        student_temp = 1,
        bins = 10,
    ):
        """
        Initialize the trainer.
        :param model: Instance of the classifier model to train.
        :param loss_fn: The loss function to evaluate with.
        :param optimizer: The optimizer to train with.
        :param device: torch.device to run training on (CPU or GPU).
        """
        super().__init__(model, device)
        self.teacher_model = teacher_model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        if self.device:
            teacher_model.to(self.device)
        self.agreement_tail = KLDivLoss(reduction='batchmean', log_target=True)
        self.softmax = Softmax(dim=1)
        self.logsoftmax = LogSoftmax(dim=1)
        self.teacher_temp = teacher_temp
        self.student_temp = student_temp
        self.bins = bins

    def train_batch(self, batch) -> BatchResult:
        X, y = batch
        if self.device:
            X = X.to(self.device)
            y = y.to(self.device)

        num_per_bin, acc_per_bin, conf_per_bin = np.zeros(self.bins, dtype=int), np.zeros(self.bins, dtype=int), np.zeros(self.bins, dtype=float)
        
        student_scores = self.model.forward(X)
        teacher_scores = self.teacher_model.forward(X)
        teacher_scores = teacher_scores.detach()
        batch_loss = self.loss_fn.forward(student_scores, teacher_scores, y)

        mask = torch.ones_like(student_scores).scatter_(1, y.unsqueeze(1), 0.)
        student_scores_no_true_class = self.logsoftmax(student_scores/self.student_temp)[mask.bool()].view(student_scores.shape[0], student_scores.shape[1]-1)
        teacher_scores_no_true_class = self.logsoftmax(teacher_scores/self.teacher_temp)[mask.bool()].view(teacher_scores.shape[0], teacher_scores.shape[1]-1)
        agreement_tail = self.agreement_tail(student_scores_no_true_class, teacher_scores_no_true_class).cpu()
        agreement_tail = agreement_tail.detach()
        student_scores_no_true_class = student_scores_no_true_class.cpu()
        teacher_scores_no_true_class = teacher_scores_no_true_class.cpu()
        
        student_scores_no_true_class_distribution = self.logsoftmax((student_scores/self.student_temp)[mask.bool()].view(student_scores.shape[0], student_scores.shape[1]-1))
        teacher_scores_no_true_class_distribution = self.logsoftmax((teacher_scores/self.teacher_temp)[mask.bool()].view(teacher_scores.shape[0], teacher_scores.shape[1]-1))
        agreement_tail_distribution = self.agreement_tail(student_scores_no_true_class_distribution, teacher_scores_no_true_class_distribution).cpu()
        agreement_tail_distribution = agreement_tail_distribution.detach()
        student_scores_no_true_class_distribution = student_scores_no_true_class_distribution.cpu()
        teacher_scores_no_true_class_distribution = teacher_scores_no_true_class_distribution.cpu()
        confidence_per_sample = torch.max(self.softmax(student_scores), dim=1)[0]
        student_confidence = confidence_per_sample.mean()

        self.optimizer.zero_grad()
        batch_loss.backward()
        batch_loss = batch_loss.item()

        self.optimizer.step()
        student_predictions = self.model.classify(X)
        teacher_predictions = self.teacher_model.classify(X)

        student_vs_true = (student_predictions == y)
        student_vs_teacher = (student_predictions == teacher_predictions) 

        for i in range(len(confidence_per_sample)):
            bin = int(np.floor(confidence_per_sample[i].item()*10))
            num_per_bin[bin] += 1
            acc_per_bin[bin] += student_vs_true[i]
            conf_per_bin[bin] += confidence_per_sample[i]

        num_correct = student_vs_true.float().sum().item()
        num_agree = student_vs_teacher.float().sum().item()
        num_agree_wrong = ((~student_vs_true)*student_vs_teacher).float().sum().item()

        student_predictions = student_predictions.cpu()
        teacher_predictions = teacher_predictions.cpu()

        student_vs_true = student_vs_true.cpu()
        student_vs_teacher = student_vs_teacher.cpu()  

        # ========================

        return BatchResult(batch_loss, num_correct, num_agree, num_agree_wrong, agreement_tail, agreement_tail_distribution, student_confidence, num_per_bin, acc_per_bin, conf_per_bin)

    def test_batch(self, batch) -> BatchResult:
        X, y = batch
        if self.device:
            X = X.to(self.device)
            y = y.to(self.device)

        num_per_bin, acc_per_bin, conf_per_bin = np.zeros(self.bins, dtype=int), np.zeros(self.bins, dtype=int), np.zeros(self.bins, dtype=float)

        self.model: Classifier
        batch_loss: float
        num_correct: int

        with torch.no_grad():
            student_scores = self.model.forward(X)
            teacher_scores = self.teacher_model.forward(X)
            batch_loss = self.loss_fn.forward(student_scores, teacher_scores, y)

            mask = torch.ones_like(student_scores).scatter_(1, y.unsqueeze(1), 0.)
            student_scores_no_true_class = self.logsoftmax(student_scores/self.student_temp)[mask.bool()].view(student_scores.shape[0], student_scores.shape[1]-1)
            teacher_scores_no_true_class = self.logsoftmax(teacher_scores/self.teacher_temp)[mask.bool()].view(teacher_scores.shape[0], teacher_scores.shape[1]-1)
            agreement_tail = self.agreement_tail(student_scores_no_true_class, teacher_scores_no_true_class)
            agreement_tail = agreement_tail.detach()
            student_scores_no_true_class = student_scores_no_true_class.cpu()
            teacher_scores_no_true_class = teacher_scores_no_true_class.cpu()
            
            student_scores_no_true_class_distribution = self.logsoftmax((student_scores/self.student_temp)[mask.bool()].view(student_scores.shape[0], student_scores.shape[1]-1))
            teacher_scores_no_true_class_distribution = self.logsoftmax((teacher_scores/self.teacher_temp)[mask.bool()].view(teacher_scores.shape[0], teacher_scores.shape[1]-1))
            agreement_tail_distribution = self.agreement_tail(student_scores_no_true_class_distribution, teacher_scores_no_true_class_distribution).cpu()
            agreement_tail_distribution = agreement_tail_distribution.detach()
            student_scores_no_true_class_distribution = student_scores_no_true_class_distribution.cpu()
            teacher_scores_no_true_class_distribution = teacher_scores_no_true_class_distribution.cpu()
            
            confidence_per_sample = torch.max(self.softmax(student_scores), dim=1)[0]
            student_confidence = confidence_per_sample.mean()

            student_predictions = self.model.classify(X)
            teacher_predictions = self.teacher_model.classify(X)

            student_vs_true = (student_predictions == y)
            student_vs_teacher = (student_predictions == teacher_predictions)

            for i in range(len(confidence_per_sample)):
                bin = int(np.floor(confidence_per_sample[i].item()*10))
                num_per_bin[bin] += 1
                acc_per_bin[bin] += student_vs_true[i]
                conf_per_bin[bin] += confidence_per_sample[i]

            num_correct = student_vs_true.float().sum().item()
            num_agree = student_vs_teacher.float().sum().item()

            student_predictions = student_predictions.cpu()
            teacher_predictions = teacher_predictions.cpu()

            student_vs_true = student_vs_true.cpu()
            student_vs_teacher = student_vs_teacher.cpu()

            num_agree_wrong = ((~student_vs_true)*student_vs_teacher).float().sum().item()
        # ========================

        return BatchResult(batch_loss, num_correct, num_agree, num_agree_wrong, agreement_tail, agreement_tail_distribution, student_confidence, num_per_bin, acc_per_bin, conf_per_bin)
