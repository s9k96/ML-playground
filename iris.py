import tensorflow.contrib.learn as skflow
from sklearn import datasets, metrics
#classification problem, 
import tensorflow as tf
iris= datasets.load_iris()
import numpy as np
# using lineraclassifier 


feature_columns = [tf.feature_column.numeric_column("x", shape=[4])]

classifier = tf.estimator.DNNClassifier(feature_columns=feature_columns,
                                          hidden_units=[10, 20, 10],
                                          n_classes=3,
                                          model_dir="/tmp/iris_model")

train_input_fn = tf.estimator.inputs.numpy_input_fn(
                                                x={"x": np.array(iris.data)},
                                                y=np.array(iris.target),
                                                num_epochs=None,
                                                shuffle=True)

classifier.train(input_fn=train_input_fn, steps=4000)
test_input_fn = tf.estimator.inputs.numpy_input_fn(
                                                x={"x": np.array(iris.data)},
                                                y=np.array(iris.target),
                                                num_epochs=1,
                                                shuffle=False)
accuracy_score = classifier.evaluate(input_fn=test_input_fn)["accuracy"]
print("\nTest Accuracy: {0:f}\n".format(accuracy_score))
