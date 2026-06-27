"""
Lightweight explainability: surface the model's top feature importances.

This uses RandomForest.feature_importances_ (global importance). It's a simple,
honest way to show WHY the model behaves as it does without pulling in heavy
dependencies like SHAP. For per-prediction local explanations you could add
SHAP later — noted as a future enhancement.
"""


def top_features(model, feature_names, k=3):
    """Return the k most important features as (name, importance) pairs."""
    importances = model.feature_importances_
    ranked = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
    return ranked[:k]
