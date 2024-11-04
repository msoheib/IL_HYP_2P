import matplotlib.pyplot as plt

def criteria_plot_population_response(pop_response, title=None):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Use viridis colormap as requested earlier
    im = ax.imshow(pop_response, aspect='auto', cmap='viridis', interpolation='nearest')
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Make left and bottom spines black
    ax.spines['left'].set_color('black')
    ax.spines['bottom'].set_color('black')
    
    # Set tick colors to black
    ax.tick_params(axis='x', colors='black')
    ax.tick_params(axis='y', colors='black')
    
    # Add colorbar
    cbar = plt.colorbar(im)
    cbar.ax.tick_params(colors='black')
    cbar.outline.set_edgecolor('black')
    
    # Set labels and title
    ax.set_xlabel('Time (ms)', fontweight='bold', color='black')
    ax.set_ylabel('Neuron', fontweight='bold', color='black')
    if title:
        ax.set_title(title, fontweight='bold', color='black')
    
    # Set background color to white
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    
    plt.tight_layout()
    return fig, ax