import shap
import torch
import numpy as np

def explain_model(model, X_sample):

    def model_predict(data):
        data = torch.tensor(data).float()
        with torch.no_grad():
            return model(data).numpy()

    explainer = shap.KernelExplainer(model_predict, X_sample)
    shap_values = explainer.shap_values(X_sample)

    shap.summary_plot(shap_values, X_sample)
