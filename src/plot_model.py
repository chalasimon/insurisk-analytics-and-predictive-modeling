import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_model_comparison(results):
    """Plot comparison of model performance."""
    # Claim severity comparison
    sev_df = pd.DataFrame(results['claim_severity']).T.reset_index()
    sev_df = sev_df.rename(columns={'index': 'model'})
    
    # Premium prediction comparison
    prem_df = pd.DataFrame(results['premium_prediction']).T.reset_index()
    prem_df = prem_df.rename(columns={'index': 'model'})
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot RMSE comparison
    sns.barplot(data=sev_df, x='model', y='rmse', ax=axes[0])
    axes[0].set_title('Claim Severity Model RMSE Comparison')
    axes[0].set_ylabel('RMSE')
    
    sns.barplot(data=prem_df, x='model', y='rmse', ax=axes[1])
    axes[1].set_title('Premium Prediction Model RMSE Comparison')
    axes[1].set_ylabel('RMSE')
    
    plt.tight_layout()
    return fig

def plot_feature_importance(importance_df, title='Feature Importance'):
    """Plot feature importance."""
    plt.figure(figsize=(10, 8))
    sns.barplot(data=importance_df.head(10), x='importance', y='feature')
    plt.title(title)
    plt.tight_layout()
    return plt.gcf()