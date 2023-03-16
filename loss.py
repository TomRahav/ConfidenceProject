import torch
from torch.nn import Module, CrossEntropyLoss, KLDivLoss, LogSoftmax

class DistillationLoss(Module):
    def __init__(self, teacher_temp = 1, student_temp = 1, alpha = 0.5, beta = 0.5):
        super().__init__()
        self.teacher_temp = teacher_temp
        self.student_temp = student_temp
        self.alpha = alpha
        self.beta = beta

    def forward(self, student_scores, teacher_scores, targets):
        student_target_loss = CrossEntropyLoss()
        student_teacher_loss = KLDivLoss(reduction='batchmean', log_target=True)
        softmax = LogSoftmax(dim=1)
        student_target_loss_output = student_target_loss((student_scores/self.student_temp), targets)
        student_teacher_loss_output = student_teacher_loss(softmax(student_scores/self.student_temp), softmax(teacher_scores/self.teacher_temp))
        return self.alpha * student_target_loss_output + self.beta * student_teacher_loss_output

    def params(self):
        return []
        
        
class DistillationLossDistribution(Module):
    def __init__(self, teacher_temp = 1, student_temp = 1, alpha = 0.5, beta = 0.5):
        super().__init__()
        self.teacher_temp = teacher_temp
        self.student_temp = student_temp
        self.alpha = alpha
        self.beta = beta

    def forward(self, student_scores, teacher_scores, targets):
        student_target_loss = CrossEntropyLoss()
        student_teacher_loss = KLDivLoss(reduction='batchmean', log_target=True)
        softmax = LogSoftmax(dim=1)
        
        mask = torch.ones_like(student_scores).scatter_(1, targets.unsqueeze(1), 0.)
        student_scores_no_true_class_distribution = softmax(((student_scores/self.student_temp)[mask.bool()]).view(student_scores.shape[0], student_scores.shape[1]-1))
        teacher_scores_no_true_class_distribution = softmax(((teacher_scores/self.teacher_temp)[mask.bool()]).view(teacher_scores.shape[0], teacher_scores.shape[1]-1))
        
        
        student_teacher_loss_output = student_teacher_loss(student_scores_no_true_class_distribution, teacher_scores_no_true_class_distribution)
        
        student_target_loss_output = student_target_loss((student_scores/self.student_temp), targets)
        
        return self.alpha * student_target_loss_output + self.beta * student_teacher_loss_output

    def params(self):
        return []        
        

class DistillationLossNoKD(Module):
    def __init__(self, student_temp = 1):
        super().__init__()
        self.student_temp = student_temp

    def forward(self, student_scores, teacher_scores, targets):
        student_target_loss = CrossEntropyLoss()
        student_target_loss_output = student_target_loss((student_scores/self.student_temp), targets)
        return student_target_loss_output

    def params(self):
        return []