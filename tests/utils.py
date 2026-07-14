import re


LOSS_PATTERN = re.compile(
    r"(?:Train: .*?\bloss|Train result: loss): ([0-9]+(?:\.[0-9]+)?)"
)
VAL_LOSS_PATTERN = re.compile(r"Test: \[\d+/\d+\] Loss ([0-9]+(?:\.[0-9]+)?)")
VAL_RESULT_PATTERN = re.compile(
    r"Val result: mIoU/mAcc/allAcc/mPrec/mRec/mF1 "
    r"(?:[0-9.]+/){2}([0-9.]+)"
)


def strip_escape_codes(text):
    return re.sub(r"\x1b(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])", "", text)


def training_losses(lines):
    losses = [
        float(match.group(1))
        for line in lines
        if (match := LOSS_PATTERN.search(line))
    ]
    assert losses, "No epoch training losses found"
    return losses


def check_loss_goes_down(lines):
    losses = training_losses(lines)
    assert losses[-1] < losses[0], f"Loss did not decrease: {losses[0]} -> {losses[-1]}"


def check_overfit(
    lines,
    max_final_loss=0.25,
    min_reduction=5.0,
    min_validation_accuracy=0.95,
):
    losses = training_losses(lines)
    assert losses[-1] <= max_final_loss
    assert losses[0] / losses[-1] >= min_reduction
    validation_losses = [
        float(match.group(1))
        for line in lines
        if (match := VAL_LOSS_PATTERN.search(line))
    ]
    validation_accuracies = [
        float(match.group(1))
        for line in lines
        if (match := VAL_RESULT_PATTERN.search(line))
    ]
    assert validation_losses, "No validation losses found"
    assert validation_accuracies, "No validation metrics found"
    assert validation_losses[-1] <= max_final_loss
    assert validation_accuracies[-1] >= min_validation_accuracy, (
        f"final validation accuracy {validation_accuracies[-1]} is below "
        f"{min_validation_accuracy}"
    )
