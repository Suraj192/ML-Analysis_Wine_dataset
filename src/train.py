from sklearn.ensemble import RandomForestClassifier
#from sklearn.feature_selection import SelectKBest, mutual_info_classif
from imblearn.over_sampling import SMOTE

def train_model(X_train, y_train):
    """
    SMOTE + Random Forest training
    """


    # ---------------------------
    # SMOTE (apply  on training data)
    # ---------------------------
    smote = SMOTE(k_neighbors=4, random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    print("Training class distribution after SMOTE:")
    print(y_resampled.value_counts())

    # ---------------------------
    # Model
    # ---------------------------
    model = RandomForestClassifier(
        n_estimators=500,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_resampled, y_resampled)

    # Return both model AND selector
    return model
