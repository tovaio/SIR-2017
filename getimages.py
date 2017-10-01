from scipy.misc import imread
import numpy as np
from sklearn import preprocessing
from sklearn import svm
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
from xgboost import XGBClassifier


files = []
for i in range(1, 925):
    i = str(i)
    i = list(i)
    z = 5 - len(i)
    for k in range(z):
        i.insert(0,'0')
    i = ''.join(i)
    files.append(i)
imgs = []
for i in range(len(files)):
    img = imread('per' + files[i] + '.ppm')
    imgs.append(img)

imgstarg = []
print(len(imgs))
for i in range(len(imgs)):
    imgstarg.append(1)


junks = []
for i in range(len(files)):
    try:
        img = imread(files[i] + '.ppm')
        junks.append(img)
    except:
        continue

junkstarg = []
for i in range(len(junks)):
    junkstarg.append(0)


imgs = np.array(imgs)
junks = np.array(junks)
imgstarg = np.array(imgstarg)
junkstarg = np.array(junkstarg)

clf = svm.SVC()
x_ = np.concatenate((imgs,junks))
y_ = np.concatenate((imgstarg,junkstarg))
X = []



'''
for i in range(len(x_)):
    img = []
    for j in range(128):
        for k in range(64):
            img.append((x_[i][j][k][0]+x_[i][j][k][1]+x_[i][j][k][2])/3) 
    X.append(img)
    print(i)
X = np.array(X)
print(X)
'''


X = pickle.load(open("pic.p", "rb"))
X = X.astype(np.float32)
y_ = y_.astype(np.float32)


seed = 7
test_size = 0.3

train_data, test_data, train_labels, test_labels = train_test_split(X, y_, test_size=test_size, random_state=seed)
'''
# fit model no training data
print('hello')
clf = XGBClassifier(n_estimators=180, max_depth=7)
clf.fit(X_train,y_train)

print("Accuracy on training set: {:.3f}".format(clf.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(clf.score(X_test, y_test)))'''


import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)


def cnn_model_fn(features, labels, mode):
    input_layer = tf.reshape(features["x"], [-1, 64, 128, 1])

    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5,5],
        padding="same",
        activation=tf.nn.relu)

    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=32,
        kernel_size=[5,5],
        padding="same",
        activation=tf.nn.relu)

    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    conv3 = tf.layers.conv2d(
        inputs=pool2,
        filters=32,
        kernel_size=[5,5],
        padding="same",
        activation=tf.nn.relu)

    pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2,2], strides=2)

    conv4 = tf.layers.conv2d(
        inputs=pool3,
        filters=32,
        kernel_size=[5,5],
        padding="same",
        activation=tf.nn.relu)

    pool4 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2,2], strides=2)
                           
    pool4_flat = tf.reshape(pool4, [-1, 4*8*32])

    dense1 = tf.layers.dense(inputs=pool4_flat, units=1024,activation=tf.nn.relu)
    dropout1 = tf.layers.dropout(
        inputs=dense1, rate=0.3, training=mode == tf.estimator.ModeKeys.TRAIN)
    dense2 = tf.layers.dense(inputs=dropout1, units=1024,activation=tf.nn.relu)
    dropout2 = tf.layers.dropout(
        inputs=dense2, rate=0.3, training=mode == tf.estimator.ModeKeys.TRAIN)
    dense3 = tf.layers.dense(inputs=dropout2, units=1024,activation=tf.nn.relu)
    dropout3 = tf.layers.dropout(
        inputs=dense3, rate=0.3, training=mode == tf.estimator.ModeKeys.TRAIN)

    logits = tf.layers.dense(inputs=dropout3, units=2)

    predictions = {
        "class": tf.argmax(logits, axis=1),
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    if mode==tf.estimator.ModeKeys.PREDICT:
        return tf.EstimatorSpec(mode=mode, predictions=predictions)

    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=2)
    loss = tf.losses.softmax_cross_entropy(
        onehot_labels=onehot_labels, logits=logits)

    if mode==tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
        labels=labels, predictions=predictions["class"])}

    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

  
def main(unused_argv):
  
    # Create the Estimator
    mnist_classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn, model_dir="/tmp/pedo_convnet_model")

    # Set up logging for predictions
    # Log the values in the "Softmax" tensor with label "probabilities"
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=50)

    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        batch_size=100,
        num_epochs=None,
        shuffle=True)
    mnist_classifier.train(
        input_fn=train_input_fn,
        steps=100,
        hooks=[logging_hook])

    # Evaluate the model and print results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": test_data},
        y=test_labels,
        num_epochs=1,
        shuffle=False)
    eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)


if __name__ == "__main__":
    tf.app.run()
