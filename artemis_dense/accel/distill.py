"""Method 10: knowledge distillation.

Built: KL(student || teacher) on logits from a LOCAL teacher checkpoint (any Artemis model with the
same vocabulary), mixed with the normal next-token loss.
Planned / approval-needed: teacher-generated text from a large open model (Llama/Qwen/...) - needs
weight downloads or API access and that model's license allowing distillation.
"""
import torch
import torch.nn.functional as F


def kd_loss(student_logits, teacher_logits, temperature: float = 2.0, mask=None):
    """Hinton KD: T^2 * KL(softmax(t/T) || softmax(s/T)), mean over (unmasked) positions."""
    T = temperature
    s = F.log_softmax(student_logits.float() / T, -1)
    t = F.log_softmax(teacher_logits.float() / T, -1)
    kl = (t.exp() * (t - s)).sum(-1)
    if mask is not None:
        kl = kl[mask]
    return kl.mean() * T * T


def distill_step_loss(student, teacher, x, y, alpha: float = 0.5, temperature: float = 2.0):
    logits, ce = student(x, y)
    with torch.no_grad():
        t_logits, _ = teacher(x)
    kd = kd_loss(logits, t_logits, temperature)
    return (1 - alpha) * ce + alpha * kd, {"ce": ce.item(), "kd": kd.item()}


def teacher_generate(*_, approved: bool = False, **__):
    raise RuntimeError("planned: teacher-generated text needs Latoya's OK (teacher weights/API + license)")
