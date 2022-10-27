import matplotlib.pyplot as plt
import numpy as np

from typing import Any, Dict, Optional, Sequence, Tuple


def imshow(images: Sequence[np.ndarray],
           figsize: Sequence[int],
           plot_settings: Optional[Sequence[Dict[str, Any]]] = None,
           layout: Optional[Tuple[int, int]] = None,
           titles: Optional[Sequence[str]] = None,
           frame: bool = True) -> plt.Figure:
    """Simplify showing one or more images
    Args:
        images: One or more images to display.
        figsize: Figure size (for the total figure, not each subplot).
        plot_settings: Dicts of kwargs for each image's call to `plt.imshow`.
        layout: (number of rows, number of columns) to arrange images in.
        titles: Axis titles.
        frame: If True, draw a frame around each image.

    Returns:
        (plt.Figure) Handle to the generated figure.

    """
    if not isinstance(images, Sequence):
        images = [images]

    if layout is None:
        layout = (1, len(images))

    if isinstance(titles, str):
        titles = [titles]

    f, axs = plt.subplots(
        nrows=layout[0],
        ncols=layout[1],
        figsize=figsize)

    if not isinstance(axs, np.ndarray):
        axs = [axs]
    for i, ax in enumerate(axs):
        if plot_settings is not None:
            ax.imshow(images[i], **plot_settings[i], extent=(0, 1, 1, 0))
        else:
            ax.imshow(images[i], extent=(0, 1, 1, 0))

        if titles is not None:
            ax.set_title(titles[i])

        ax.axis('tight')
        if frame:
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])
        else:
            ax.axis('off')

    return f
