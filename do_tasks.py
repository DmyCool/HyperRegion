import json
import numpy as np
from sklearn.cluster import KMeans
from sklearn import linear_model
from sklearn.model_selection import KFold

from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error



crime_count_label = np.load("./data/raw_data/crime_counts.npy")[:, 0]
check_in_label = np.load("./data/raw_data/check_counts.npy")

cd = json.load(open("./data/raw_data/mh_cd.json"))
cd_labels = np.zeros((180))
for i in range(180):
    cd_labels[i] = cd[str(i)]


def compute_metrics(y_pred, y_test):
    y_pred[y_pred<0] = 0
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mae, np.sqrt(mse), r2


def regression(X_train, y_train, X_test, alpha):
    reg = linear_model.Ridge(alpha=alpha)
    X_train = np.array(X_train, dtype=float)
    y_train = np.array(y_train, dtype=float)
    reg.fit(X_train, y_train)

    y_pred = reg.predict(X_test)
    return y_pred

def kf_predict(X, Y):
    kf = KFold(n_splits=5)
    y_preds = []
    y_truths = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
        y_pred = regression(X_train, y_train, X_test, 1)
        y_preds.append(y_pred)
        y_truths.append(y_test)

    return np.concatenate(y_preds), np.concatenate(y_truths)


def predict_regression(embs, labels):
    y_pred, y_test = kf_predict(embs, labels)
    mae, rmse, r2 = compute_metrics(y_pred, y_test)

    # print("MAE: ", mae)
    # print("RMSE: ", rmse)
    # print("R2: ", r2)

    return mae, rmse, r2


def lu_classify(emb, cd_labels):
    n = 12
    kmeans = KMeans(n_clusters=n, random_state=3)
    emb_labels = kmeans.fit_predict(emb)

    nmi = normalized_mutual_info_score(cd_labels, emb_labels)
    ars = adjusted_rand_score(cd_labels, emb_labels)

    # print("emb nmi: {:.3f}".format(nmi))
    # print("emb ars: {:.3f}".format(ars))

    return nmi, ars


def do_tasks(embs):

    # print("Crime Count Prediction: ")
    crime_mae, crime_rmse, crime_r2 = predict_regression(embs, crime_count_label)

    # print("Check-in Prediction: ")
    check_mae, check_rmse, check_r2 = predict_regression(embs, check_in_label)

    # print("Land Usage Prediction: ")
    nmi, ars = lu_classify(embs, cd_labels)

    return crime_mae, crime_rmse, crime_r2, check_mae, check_rmse, check_r2, nmi, ars


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore")

    emb = np.load(f'./data/embeddings/emb_4_214.npy')
    crime_mae, crime_rmse, crime_r2, check_mae, check_rmse, check_r2, nmi, ars = do_tasks(emb)
    print(crime_mae, crime_rmse, crime_r2, check_mae, check_rmse, check_r2, nmi, ars)

