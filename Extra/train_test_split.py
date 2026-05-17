# train_test_split.py

from sklearn.model_selection import train_test_split

def split_data(X_img, X_lm, y, test_size=0.2):
    X_img_train, X_img_test, X_lm_train, X_lm_test, y_train, y_test = train_test_split(
        X_img, X_lm, y, test_size=test_size, stratify=y, random_state=42
    )

    return X_img_train, X_img_test, X_lm_train, X_lm_test, y_train, y_test