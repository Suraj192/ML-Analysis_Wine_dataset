from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from imblearn.over_sampling import SMOTE

def train_model(X_train, y_train):
    """
    Feature selection + SMOTE + Random Forest training
    """

    # ---------------------------
    # Feature Selection (fit ONLY on training data)
    # ---------------------------
    selector = SelectKBest(score_func=mutual_info_classif, k=6)
    X_train_selected = selector.fit_transform(X_train, y_train)

    # ---------------------------
    # SMOTE (apply ONLY on training data)
    # ---------------------------
    smote = SMOTE(k_neighbors=4, random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train_selected, y_train)

    print("Training class distribution after SMOTE:")
    print(y_resampled.value_counts())

    # ---------------------------
    # Model
    # ---------------------------
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_resampled, y_resampled)

    # Return both model AND selector
    return model, selector
