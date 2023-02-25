import numpy as np
import json
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight
from sklearn.metrics import balanced_accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn import svm
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.model_selection import train_test_split
from glmnet import LogitNet

MIN_TRIALS_PER_CLASS = 5

def compute_weights_by_class(target):
    # Compute weights for each target class. This is especially important for confidence data, which are highly unbalanced.
    # For example if we had 2 classes and there were 50 trials of class 0 and 100 trials of class 2, then the weights would
    # be (0.666, 0.333), i.e. the class with half as many trials would be weighted twice as much when fitting the network.
    utarget_sub = np.unique(target)
    weights = class_weight.compute_class_weight("balanced", classes=utarget_sub, y=target)
    weights = dict(zip(utarget_sub, weights/np.sum(weights)))
    return weights

def compute_weights_by_sample(target):
    # Like compute_weights_by_class, but return a weight for each sample.
    weights_class = compute_weights_by_class(target)
    weights_sample = np.full((len(target)), np.nan)
    for i, t in enumerate(target):
        weights_sample[i] = weights_class[t]
    return weights_sample

def train_conv(target):
    # Train on the raw data (not dim reduced).
    data = np.load("train.npz", allow_pickle=True)
    with open("train_split.json", "r") as f:
        split = json.load(f)

    # The number of output neurons at the last layer is equal to the number of categories we're classifying into.
    if target == "resp":
        noutput = 2
    elif target == "conf":
        noutput = 3
    elif target == "condition":
        noutput = 5

    nsub = np.max(data['sub']) + 1

    scores = np.full((nsub), np.nan)
    for sno in range(nsub):
        # Find this participant's data.
        input_sub = data['input'][data['sub'] == sno, :, :]
        target_sub = data[target][data['sub'] == sno]

        # Convert string labels to int.
        if target == "condition":
            _, target_sub = np.unique(target_sub, return_inverse=True)

        input_sub_train = input_sub[split['all_idx_train'][sno], :, :]
        input_sub_valid = input_sub[split['all_idx_valid'][sno], :, :]
        target_sub_train = target_sub[split['all_idx_train'][sno]]
        target_sub_valid = target_sub[split['all_idx_valid'][sno]]

        if np.unique(target_sub_train).size < noutput or np.any(np.unique(target_sub_train, return_counts=True)[1] < MIN_TRIALS_PER_CLASS):
            continue

        # Standardize predictors.
        train_shape = input_sub_train.shape
        valid_shape = input_sub_valid.shape
        input_sub_train = np.reshape(input_sub_train, (input_sub_train.shape[0], input_sub_train.shape[1]*input_sub_train.shape[2]), order="C")
        input_sub_valid = np.reshape(input_sub_valid, (input_sub_valid.shape[0], input_sub_valid.shape[1]*input_sub_valid.shape[2]), order="C")
        scale = StandardScaler()
        scale = scale.fit(input_sub_train)
        input_sub_train = scale.transform(input_sub_train)
        input_sub_valid = scale.transform(input_sub_valid)
        input_sub_train = np.reshape(input_sub_train, train_shape, order="C")
        input_sub_valid = np.reshape(input_sub_valid, valid_shape, order="C")

        input_sub_train[np.isnan(input_sub_train)] = 0
        input_sub_valid[np.isnan(input_sub_valid)] = 0

        weights = compute_weights_by_class(target_sub_train)
        weights_valid_sample = compute_weights_by_sample(target_sub_valid)

        # Build the network.
        inputs = keras.Input(shape=input_sub.shape[1:3])
        x = inputs
        x = keras.layers.Conv1D(filters=5, kernel_size=9, activation="relu")(x)  
        x = keras.layers.MaxPooling1D(3)(x) 
        x = keras.layers.Conv1D(filters=5, kernel_size=9, activation="relu")(x)
        x = keras.layers.MaxPooling1D(3)(x)
        x = keras.layers.Conv1D(filters=5, kernel_size=9, activation="relu")(x)
        x = keras.layers.MaxPooling1D(3)(x)
        x = keras.layers.Flatten()(x)
        x = keras.layers.Dense(30, activation="relu")(x) 
        x = keras.layers.Dropout(0.5)(x)
        outputs = keras.layers.Dense(noutput, activation="softmax")(x)

        # Fit the network and store weighted accuracy for the validation data from the last epoch. 
        model = keras.Model(inputs, outputs)
        model.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.SparseCategoricalCrossentropy(), weighted_metrics=[keras.metrics.SparseCategoricalAccuracy()])
        history = model.fit(x=input_sub_train, y=target_sub_train, verbose=2, class_weight=weights, validation_data=(input_sub_valid, target_sub_valid, weights_valid_sample), batch_size=32, epochs=100)
        scores[sno] = history.history['val_sparse_categorical_accuracy'][-1]

    print('Balanced accuracy (N=%d): %.3f (%.3f)' % (np.sum(~np.isnan(scores)), np.nanmean(scores), np.nanstd(scores)))

def test_classifier(classifier, target, file_train, normalize):
    data = np.load(file_train, allow_pickle=True)
    with open("train_split.json", "r") as f:
        split = json.load(f)

    input_key = 'input' if 'input' in data else 'reduced_input'
    input = data[input_key]
    if input_key == 'input':
        input = np.moveaxis(input, -1, 1)
        input = np.reshape(input, (input.shape[0], input.shape[1]*input.shape[2]), order="C")

    nsub = np.max(data['sub']) + 1

    scores = np.full((nsub), np.nan)
    for sno in range(nsub):
        input_sub = input[data['sub'] == sno, :]
        target_sub = data[target][data['sub'] == sno]

        input_sub_train = input_sub[split['all_idx_train'][sno], :]
        input_sub_valid = input_sub[split['all_idx_valid'][sno], :]
        target_sub_train = target_sub[split['all_idx_train'][sno]]
        target_sub_valid = target_sub[split['all_idx_valid'][sno]]

        if normalize:
            scale = StandardScaler()
            scale = scale.fit(input_sub_train)
            input_sub_train = scale.transform(input_sub_train)
            input_sub_valid = scale.transform(input_sub_valid)

        input_sub_train[np.isnan(input_sub_train)] = 0
        input_sub_valid[np.isnan(input_sub_valid)] = 0

        if target == "resp":
            noutput = 2
        elif target == "conf":
            noutput = 3
        elif target == "condition":
            noutput = 5
        if np.unique(target_sub_train).size < noutput or np.any(np.unique(target_sub_train, return_counts=True)[1] < MIN_TRIALS_PER_CLASS):
            continue

        weights = compute_weights_by_sample(target_sub_train)

        model = classifier.fit(input_sub_train, target_sub_train, weights)
        scores[sno] = balanced_accuracy_score(target_sub_valid, model.predict(input_sub_valid))

    print('Balanced accuracy (N=%d): %.3f (%.3f)' % (np.sum(~np.isnan(scores)), np.nanmean(scores), np.nanstd(scores)))

def random_forest(target, file_train):
    test_classifier(RandomForestClassifier(n_jobs=64, n_estimators=100, max_depth=None), target, file_train, False)

def boosting(target, file_train, n_estimators=100, max_depth=3):
    test_classifier(GradientBoostingClassifier(n_estimators=n_estimators, max_depth=max_depth), target, file_train, False)

def svmc(target, file_train, kernel, C=1):
    test_classifier(svm.SVC(kernel=kernel, C=C), target, file_train, True)

def elastic_net(target, file_train):
    data = np.load(file_train, allow_pickle=True)

    input_key = 'input' if 'input' in data else 'reduced_input'
    input = data[input_key]
    if input_key == 'input':
        input = np.moveaxis(input, -1, 1)
        input = np.reshape(input, (input.shape[0], input.shape[1]*input.shape[2]), order="C")

    X_train, X_valid, y_train, y_valid = train_test_split(input, data[target], test_size=0.2)

    scale = StandardScaler()
    scale = scale.fit(X_train)
    X_train = scale.transform(X_train)
    X_valid = scale.transform(X_valid)

    X_train[np.isnan(X_train)] = 0
    X_valid[np.isnan(X_valid)] = 0

    weights = compute_weights_by_sample(y_train)

    logitnet = LogitNet(alpha=1)
    skf = StratifiedKFold(n_splits=5, shuffle=True)
    classifier = GridSearchCV(logitnet, {"alpha": [0.05, 0.5, 0.95]}, scoring="balanced_accuracy", cv=skf, verbose=3)
    model = classifier.fit(X_train, y_train, sample_weight=weights)
    prediction = model.predict(X_valid)

    print('Balanced accuracy: %0.3f' % balanced_accuracy_score(y_valid, prediction))

#elastic_net("resp", "train.npz")      
random_forest("resp", "train_pca.npz")                                                                     