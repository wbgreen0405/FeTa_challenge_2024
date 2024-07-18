import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.express as px
from sklearn.preprocessing import StandardScaler

def plot_predictions(y_val, y_pred, scaler):
    y_pred = scaler.inverse_transform(y_pred)
    y_val = scaler.inverse_transform(y_val)

    # Scatter plot of actual vs predicted measurements using Plotly
    fig = go.Figure()

    # Adding scatter traces for each measurement
    measurements = ['LCC', 'HV', 'bBIP', 'sBIP', 'TCD']
    colors = px.colors.qualitative.Plotly

    for i in range(5):
        fig.add_trace(go.Scatter(
            x=y_val[:, i], y=y_pred[:, i],
            mode='markers',
            name=measurements[i],
            marker=dict(color=colors[i])
        ))

    # Adding the diagonal line
    fig.add_trace(go.Scatter(
        x=[y_val.min(), y_val.max()],
        y=[y_val.min(), y_val.max()],
        mode='lines',
        line=dict(color='black', dash='dash'),
        showlegend=False
    ))

    fig.update_layout(
        title='Actual vs Predicted Measurements',
        xaxis_title='Actual Measurements',
        yaxis_title='Predicted Measurements',
        legend_title='Measurements',
        width=800,
        height=600
    )

    fig.show()

    # Plot predicted vs actual images with measurements on top
    num_images = 5
    fig, axes = plt.subplots(num_images, 2, figsize=(10, num_images * 5))
    for i in range(num_images):
        middle_slice = X_val[i, :, :, 64, 0]

        axes[i, 0].imshow(middle_slice, cmap='gray')
        axes[i, 0].set_title(f'Actual: LCC={y_val[i, 0]:.2f}, HV={y_val[i, 1]:.2f}, bBIP={y_val[i, 2]:.2f}, sBIP={y_val[i, 3]:.2f}, TCD={y_val[i, 4]:.2f}', fontsize=8)
        axes[i, 0].axis('off')

        axes[i, 1].imshow(middle_slice, cmap='gray')
        axes[i, 1].set_title(f'Predicted: LCC={y_pred[i, 0]:.2f}, HV={y_pred[i, 1]:.2f}, bBIP={y_pred[i, 2]:.2f}, sBIP={y_pred[i, 3]:.2f}, TCD={y_pred[i, 4]:.2f}', fontsize=8)
        axes[i, 1].axis('off')

    plt.tight_layout()
    plt.show()
