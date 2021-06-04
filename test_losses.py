import tensorflow as tf
from dpr.losses import HardNegVsNegDPRLoss, HardNegVsNegSoftMaxDPRLoss, ThreeLevelDPRLoss, TwoLevelDPRLoss


def main():
    dim_size = 768
    batch_size = 8
    within_size = 4
    loss_fn = HardNegVsNegSoftMaxDPRLoss(batch_size=batch_size, within_size=within_size)
    q_tensors = tf.random.normal([batch_size, dim_size])
    ctx_tensors = tf.random.normal([batch_size * within_size, dim_size])

    q_tensors = tf.Variable(q_tensors)
    ctx_tensors = tf.Variable(ctx_tensors)
    with tf.GradientTape(persistent=True) as tape:
        nll_within_loss, hardnegvsneg_loss = loss_fn(q_tensors, ctx_tensors)

    grads_1 = tape.gradient(nll_within_loss, q_tensors)
    grads_2 = tape.gradient(hardnegvsneg_loss, q_tensors)

    print("done")


if __name__ == "__main__":
    main()