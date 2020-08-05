import tensorflow as tf
from semmatch.utils.exception import ConfigureError


def rank_hinge_loss(labels, logits, params):
    num_retrieval = params.get('num_retrieval', None)
    if num_retrieval is None:
        raise ConfigureError("The parameter num_retrieval is not assigned or the dataset is not support rank loss.")
    margin = params.get('rank_loss_margin', 1.0)
    labels = tf.argmax(labels, axis=-1)
    labels = tf.reshape(labels, (-1, num_retrieval))
    logits = tf.reshape(logits, (-1, num_retrieval))
    label_mask = tf.cast(tf.sign(labels), tf.float32)
    label_count = tf.reduce_sum(label_mask, axis=-1)
    y_pos = tf.reduce_sum(label_mask * logits, axis=-1)/label_count
    y_neg = tf.reduce_sum((1.-label_mask) * logits, axis=-1)/(num_retrieval-label_count)
    loss = tf.maximum(0., margin-y_pos+y_neg)
    loss = tf.reduce_mean(loss)
    return loss


def multilabel_categorical_crossentropy(labels, logits):
    """多标签分类的交叉熵
    说明：y_true和y_pred的shape一致，y_true的元素非0即1，
         1表示对应的类为目标类，0表示对应的类为非目标类。
    """
    y_pred = (1 - 2 * labels) * logits
    y_pred_neg = y_pred - labels * 1e12
    y_pred_pos = y_pred - (1 - labels) * 1e12
    zeros = tf.zeros_like(y_pred[..., :1])
    y_pred_neg = tf.concat([y_pred_neg, zeros], axis=-1)
    y_pred_pos = tf.concat([y_pred_pos, zeros], axis=-1)
    neg_loss = tf.reduce_logsumexp(y_pred_neg, axis=-1)
    pos_loss = tf.reduce_logsumexp(y_pred_pos, axis=-1)
    return tf.reduce_mean(neg_loss + pos_loss)


def multilabel_categorical_crossentropy_topk(labels, logits):
    """多标签分类的交叉熵
    说明：y_true和y_pred的shape一致，y_true的元素非0即1，
         1表示对应的类为目标类，0表示对应的类为非目标类。
    """
    y_pred = (1 - 2 * labels) * logits
    y_pred_neg = y_pred - labels * 1e12
    y_pred_pos = y_pred - (1 - labels) * 1e12

    y_pred_neg = tf.reduce_sum(tf.exp(y_pred_neg))
    y_pred_pos = tf.reduce_sum(tf.exp(y_pred_pos))
    return tf.log(1+y_pred_neg*y_pred_pos)


def focal_loss(logits, labels, alpha=1.0, epsilon=1e-9, gamma=2.0):
    prob = tf.nn.softmax(logits, axis=-1)  # bs, cn
    prob = tf.clip_by_value(prob, epsilon, 1 - epsilon)
    loss = -tf.reduce_mean(alpha * (1 - prob) ** gamma * tf.log(prob) * labels)
    return loss

    # neg_loss = tf.reduce_logsumexp(y_pred_neg, axis=-1)
    # pos_loss = tf.reduce_logsumexp(y_pred_pos, axis=-1)
    # return neg_loss + pos_loss


class GHM_Loss:
    def __init__(self, bins=10, momentum=0.75):
        self.g =None
        self.bins = bins
        self.momentum = momentum
        self.valid_bins = tf.constant(0.0, dtype=tf.float32)
        self.edges_left, self.edges_right = self.get_edges(self.bins)
        if momentum > 0:
            acc_sum = [0.0 for _ in range(bins)]
            self.acc_sum = tf.Variable(acc_sum, trainable=False)

    @staticmethod
    def get_edges(bins):
        edges_left = [float(x) / bins for x in range(bins)]
        edges_left = tf.constant(edges_left)  # [bins]
        edges_left = tf.expand_dims(edges_left, -1)  # [bins, 1]
        edges_left = tf.expand_dims(edges_left, -1)  # [bins, 1, 1]
        edges_left = tf.expand_dims(edges_left, -1)  # [bins, 1, 1, 1]

        edges_right = [float(x) / bins for x in range(1, bins + 1)]
        edges_right[-1] += 1e-3
        edges_right = tf.constant(edges_right)  # [bins]
        edges_right = tf.expand_dims(edges_right, -1)  # [bins, 1]
        edges_right = tf.expand_dims(edges_right, -1)  # [bins, 1, 1]
        edges_right = tf.expand_dims(edges_right, -1)  # [bins, 1, 1, 1]
        return edges_left, edges_right

    def calc(self, g, valid_mask):
        edges_left, edges_right = self.edges_left, self.edges_right
        alpha = self.momentum
        # valid_mask = tf.cast(valid_mask, dtype=tf.bool)

        tot = tf.maximum(tf.reduce_sum(tf.cast(valid_mask, dtype=tf.float32)), 1.0)
        inds_mask = tf.logical_and(tf.greater_equal(g, edges_left), tf.less(g, edges_right))
        zero_matrix = tf.cast(tf.zeros_like(inds_mask), dtype=tf.float32)  # [bins, batch_num, class_num]

        inds = tf.cast(tf.logical_and(inds_mask, valid_mask), dtype=tf.float32)  # [bins, batch_num, class_num]

        num_in_bin = tf.reduce_sum(inds, axis=[1, 2, 3])  # [bins]
        valid_bins = tf.greater(num_in_bin, 0)  # [bins]

        num_valid_bin = tf.reduce_sum(tf.cast(valid_bins, dtype=tf.float32))

        if alpha > 0:
            update = tf.assign(self.acc_sum,
                               tf.where(valid_bins, alpha * self.acc_sum + (1 - alpha) * num_in_bin, self.acc_sum))
            with tf.control_dependencies([update]):
                acc_sum_tmp = tf.identity(self.acc_sum, name='updated_accsum')
                acc_sum = tf.expand_dims(acc_sum_tmp, -1)  # [bins, 1]
                acc_sum = tf.expand_dims(acc_sum, -1)  # [bins, 1, 1]
                acc_sum = tf.expand_dims(acc_sum, -1)  # [bins, 1, 1, 1]
                acc_sum = acc_sum + zero_matrix  # [bins, batch_num, class_num]
                weights = tf.where(tf.equal(inds, 1), tot / acc_sum, zero_matrix)
                weights = tf.reduce_sum(weights, axis=0)

        else:
            num_in_bin = tf.expand_dims(num_in_bin, -1)  # [bins, 1]
            num_in_bin = tf.expand_dims(num_in_bin, -1)  # [bins, 1, 1]
            num_in_bin = tf.expand_dims(num_in_bin, -1)  # [bins, 1, 1, 1]
            num_in_bin = num_in_bin + zero_matrix  # [bins, batch_num, class_num]
            weights = tf.where(tf.equal(inds, 1), tot / num_in_bin, zero_matrix)
            weights = tf.reduce_sum(weights, axis=0)

        weights = weights / num_valid_bin

        return weights, tot

    def ghm_class_loss(self, logits, targets, masks=None):
        """ Args:
        input [batch_num, class_num]:
            The direct prediction of classification fc layer.
        target [batch_num, class_num]:
            Binary target (0 or 1) for each sample each class. The value is -1
            when the sample is ignored.
        """
        train_mask = (1 - tf.cast(tf.equal(targets, -1), dtype=tf.float32))
        self.g = tf.abs(tf.sigmoid(logits) - targets) # [batch_num, class_num]
        g = tf.expand_dims(self.g, axis=0)  # [1, batch_num, class_num]

        if masks is None:
            masks = tf.ones_like(targets)
        valid_mask = masks > 0
        weights, tot = self.calc(g, valid_mask)
        print(weights.shape)
        ghm_class_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=targets*train_mask,
                                                                 logits=logits)
        ghm_class_loss = tf.reduce_sum(ghm_class_loss * weights) / tot

        return ghm_class_loss

    def ghm_regression_loss(self, logits, targets, masks):
        """ Args:
        input [batch_num, *(* class_num)]:
            The prediction of box regression layer. Channel number can be 4 or
            (4 * class_num) depending on whether it is class-agnostic.
        target [batch_num,  *(* class_num)]:
            The target regression values with the same size of input.
        """
        mu = self.mu

        # ASL1 loss
        diff = logits - targets
        # gradient length
        g = tf.abs(diff / tf.sqrt(mu * mu + diff * diff))

        if masks is None:
            masks = tf.ones_like(targets)
        valid_mask = masks > 0

        weights, tot = self.calc(g, valid_mask)

        ghm_reg_loss = tf.sqrt(diff * diff + mu * mu) - mu
        ghm_reg_loss = tf.reduce_sum(ghm_reg_loss * weights) / tot

        return ghm_reg_loss


class GHM_Loss2:
    def __init__(self, bins=10, momentum=0.75):
        self.g = None
        self.bins = bins
        self.momentum = momentum
        self.valid_bins = tf.constant(0.0, dtype=tf.float32)
        self.edges_left, self.edges_right = self.get_edges(self.bins)
        if momentum > 0:
            acc_sum = [0.0 for _ in range(bins)]
            self.acc_sum = tf.Variable(acc_sum, trainable=False)

    @staticmethod
    def get_edges(bins):
        edges_left = [float(x) / bins for x in range(bins)]
        edges_left = tf.constant(edges_left) # [bins]
        edges_left = tf.expand_dims(edges_left, -1) # [bins, 1]
        edges_left = tf.expand_dims(edges_left, -1) # [bins, 1, 1]

        edges_right = [float(x) / bins for x in range(1, bins + 1)]
        edges_right[-1] += 1e-6
        edges_right = tf.constant(edges_right) # [bins]
        edges_right = tf.expand_dims(edges_right, -1) # [bins, 1]
        edges_right = tf.expand_dims(edges_right, -1) # [bins, 1, 1]
        return edges_left, edges_right

    def calc(self, g, valid_mask):
        edges_left, edges_right = self.edges_left, self.edges_right
        alpha = self.momentum
        # valid_mask = tf.cast(valid_mask, dtype=tf.bool)

        tot = tf.maximum(tf.reduce_sum(tf.cast(valid_mask, dtype=tf.float32)), 1.0)
        inds_mask = tf.logical_and(tf.greater_equal(g, edges_left), tf.less(g, edges_right))
        zero_matrix = tf.cast(tf.zeros_like(inds_mask), dtype=tf.float32)  # [bins, batch_num, class_num]

        inds = tf.cast(tf.logical_and(inds_mask, valid_mask), dtype=tf.float32)  # [bins, batch_num, class_num]

        num_in_bin = tf.reduce_sum(inds, axis=[1, 2])  # [bins]
        valid_bins = tf.greater(num_in_bin, 0)  # [bins]

        num_valid_bin = tf.reduce_sum(tf.cast(valid_bins, dtype=tf.float32))

        if alpha > 0:
            update = tf.assign(self.acc_sum,
                               tf.where(valid_bins, alpha * self.acc_sum + (1 - alpha) * num_in_bin, self.acc_sum))
            with tf.control_dependencies([update]):
                acc_sum_tmp = tf.identity(self.acc_sum, name='updated_accsum')
                acc_sum = tf.expand_dims(acc_sum_tmp, -1)  # [bins, 1]
                acc_sum = tf.expand_dims(acc_sum, -1)  # [bins, 1, 1]
                acc_sum = acc_sum + zero_matrix  # [bins, batch_num, class_num]
                weights = tf.where(tf.equal(inds, 1), tot / acc_sum, zero_matrix)
                weights = tf.reduce_sum(weights, axis=0)

        else:
            num_in_bin = tf.expand_dims(num_in_bin, -1)  # [bins, 1]
            num_in_bin = tf.expand_dims(num_in_bin, -1)  # [bins, 1, 1]
            num_in_bin = num_in_bin + zero_matrix  # [bins, batch_num, class_num]
            weights = tf.where(tf.equal(inds, 1), tot / num_in_bin, zero_matrix)
            weights = tf.reduce_sum(weights, axis=0)

        weights = weights / num_valid_bin

        return weights, tot

    def ghm_class_loss(self, logits, targets, masks=None):
        """ Args:
        input [batch_num, class_num]:
            The direct prediction of classification fc layer.
        target [batch_num, class_num]:
            Binary target (0 or 1) for each sample each class. The value is -1
            when the sample is ignored.
        """
        train_mask = (1 - tf.cast(tf.equal(targets, -1), dtype=tf.float32))
        self.g = tf.abs(tf.sigmoid(logits) - targets) # [batch_num, class_num]
        g = tf.expand_dims(self.g, axis=0)  # [1, batch_num, class_num]

        if masks is None:
            masks = tf.ones_like(targets)
        valid_mask = masks > 0
        weights, tot = self.calc(g, valid_mask)
        print(weights.shape)
        ghm_class_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=targets*train_mask,
                                                                 logits=logits)
        ghm_class_loss = tf.reduce_sum(ghm_class_loss * weights) / tot

        return ghm_class_loss

    def ghm_regression_loss(self, logits, targets, masks):
        """ Args:
        input [batch_num, *(* class_num)]:
            The prediction of box regression layer. Channel number can be 4 or
            (4 * class_num) depending on whether it is class-agnostic.
        target [batch_num,  *(* class_num)]:
            The target regression values with the same size of input.
        """
        mu = self.mu

        # ASL1 loss
        diff = logits - targets
        # gradient length
        g = tf.abs(diff / tf.sqrt(mu * mu + diff * diff))

        if masks is None:
            masks = tf.ones_like(targets)
        valid_mask = masks > 0

        weights, tot = self.calc(g, valid_mask)

        ghm_reg_loss = tf.sqrt(diff * diff + mu * mu) - mu
        ghm_reg_loss = tf.reduce_sum(ghm_reg_loss * weights) / tot

        return ghm_reg_loss