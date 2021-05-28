import tensorflow as tf
from model import create_model
from dataset import Dataset
from configs import *
from utils import *

import shutil
import datetime

model = create_model()

dataset = Dataset()
model.summary()


if TENSORBOARD:
    if DELETE_LOGS:
        try: shutil.rmtree('logs')
        except FileNotFoundError: pass
    train_log_dir = 'logs/' + str(datetime.datetime.now().time().replace(microsecond=0)).replace(':', '_') + '_model_train'
    test_log_dir  = 'logs/' + str(datetime.datetime.now().time().replace(microsecond=0)).replace(':', '_') + '_model_test'

    train_writer = tf.summary.create_file_writer(logdir=train_log_dir)
    test_writer  = tf.summary.create_file_writer(logdir=test_log_dir)

    def write_summary():
        tf.summary.scalar('loss',  metrics_loss.result(),  step=step)
        tf.summary.scalar('error', metrics_error.result(), step=step)


optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
loss_obj  = tf.keras.losses.MeanSquaredError()
metrics_loss  = tf.keras.metrics.Mean(name='train loss')
metrics_error = tf.keras.metrics.MeanAbsoluteError(name='train_error')

step = 0
for epoch_num in range(EPOCHS):

    for X, y in dataset.train_ds:
        step += 1
        with tf.GradientTape() as tape:
            pred = model(X, training=True)
            loss = loss_obj(y, pred)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        metrics_loss.reset_states()
        metrics_error.reset_states()

        metrics_loss(loss)
        metrics_error(y, pred)

        print(y.round(3))
        print(pred.numpy().round(3))
        train_stats(epoch_num, EPOCHS,
                    dataset.train_ds.batch_count, dataset.train_ds.num_batches,
                    step=step,
                    loss=round(float(metrics_loss.result()), 3),
                    error=round(float(metrics_error.result()), 3)
                    )
        if TENSORBOARD:
            with train_writer.as_default(): write_summary()

    metrics_loss.reset_states()
    metrics_error.reset_states()

    for X, y in dataset.test_ds:
        pred = model(X, training=True)
        loss = loss_obj(y, pred)

        metrics_loss(loss)
        metrics_error(y, pred)

    if TENSORBOARD:
        with test_writer.as_default(): write_summary()

model.save(MODEL_SAVE_DIR)

