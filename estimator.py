import argparse
import tensorflow as tf

import iris_data


parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=100, type=int, help='batch size')
parser.add_argument('--train_steps', default=1000, type=int,
                    help='number of training steps')

def main(argv):
    args = parser.parse_args(argv[1:])

    # Fetch the data
    # train_x.keys() are the CSV_COLUMN_NAMES GIVEN IN iris_data.py file
    # train_x are the sizes of the petals training values without the last 
    #     column that indicates the type of the iris
    # train_y is the last column that indicates the type of the iris
    # test_x.keys(), test_x, test_y same that train_x and train_y
    (train_x, train_y), (test_x, test_y) = iris_data.load_data()

    # Feature columns describe how to use the input.
    # need to specify which the feature columns are to Tensor flow
    # gets the info, name of the feature and the value and init tf stuff
    # ie: _NumericColumn(key='SepalLength', shape=(1,), default_value=None, dtype=tf.float32
    my_feature_columns = []
    for key in train_x.keys():
        my_feature_columns.append(tf.feature_column.numeric_column(key=key))


    print(my_feature_columns)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
