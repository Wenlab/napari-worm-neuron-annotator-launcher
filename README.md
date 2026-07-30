# Worm Neuron Annotator Usage Guide

This repository provides a reproducible launcher for opening compatible volumetric imaging datasets with napari plugin [`napari-worm-neuron-annotator`](https://pypi.org/project/napari-worm-neuron-annotator/). It loads an Image array, a neuron Labels array, and read-only neuron bounding-box ROIs, then opens the annotator directly in napari.

The launcher script `launch_datasets.py` creates and docks the plugin automatically, so the plugin does not need to be opened manually from napari's **Plugins** menu.

## Interface preview

![Interface](assets/interface.png)

## Prerequisites

- [Pixi](https://pixi.sh/latest/installation/): Pixi is the recommended installation method because it provides a consistent, locked Python, napari, Qt, and plugin environment across supported platforms.
- A compatible dataset containing the three NPY files described below:

```text
dataset/
├── volumes.npy
├── neuron_mask.npy
└── neuron_point_tuple.npy
```

| File                     | Expected content                                                                                                                                              |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `volumes.npy`            | Image array with shape `(t, z, y, x)`                                                                                                                         |
| `neuron_point_tuple.npy` | ROI array with shape `(T, N, K)`, where `K >= 6`. The first six fields are `x`, `y`, `z_scaled`, `width`, `height`, and `depth_scaled`.                          |
| `neuron_mask.npy`        | Integer Labels array with the same `(t, z, y, x)` shape as `volumes.npy`; in ROI mode, `label_value = neuron_id + 1`.                                         |

## Quick start

### 1. Clone the repository

```bash
git clone <repository-url>
cd napari-worm-neuron-annotator-launcher
```

Use Pixi to create the environment:

```shell
pixi install
```

### 2. Configure the dataset path

Open `script/launch_datasets.py` and edit `DATA_DIR` near the top of the file.

### 3. Start the application

Run this command from the repository root:

```bash
pixi run start
```

## Launcher script configuration

The main dataset-specific settings are located near the top of `script/launch_datasets.py`:
```python

DATA_DIR = Path("/path/to/dataset")

Z_DIVISOR = 5.0
LAYER_SCALE_TZYX = (1.0, 5.0, 1.0, 1.0)
IMAGE_CONTRAST_LIMITS = (102, 400)
```

| Parameter               | Meaning                                                       |
| ----------------------- | ------------------------------------------------------------- |
| `DATA_DIR`              | Directory containing the three input NPY files                |
| `Z_DIVISOR`             | Converts scaled Z coordinates and depths into image Z indices |
| `LAYER_SCALE_TZYX`      | napari layer scale in `(t, z, y, x)` order                    |
| `IMAGE_CONTRAST_LIMITS` | Initial Image display contrast range                          |

`IMAGE_CONTRAST_LIMITS` affects visualization only. The contrast limits can be adjusted later in the Image layer controls. Gamma is a separate display parameter.

![Image Contrast](assets/image_contrast.png)

`Z_DIVISOR` and the Z component of `LAYER_SCALE_TZYX` serve different purposes. The former converts ROI coordinates to array indices; the latter defines napari world coordinates. Do not apply the layer Z scale to the ROI coordinates a second time.

## What happens at startup

The launcher:

1. Checks that all three input files exist.
2. Opens Image and Labels arrays using read-only NumPy memory mapping.
3. Verifies matching four-dimensional `(t, z, y, x)` shapes.
4. Verifies that the Labels array has an integer data type.
5. Creates the napari viewer and adds the Image and Labels layers.
6. Creates and docks the plugin.
7. Loads the ROI file.
8. Checks, activates, and locates the first valid neuron.

## Basic usage

### Neuron selection

- Clicking a neuron row checks and activates that neuron without clearing existing checks.
- Checking a checkbox adds and activates that neuron.
- Unchecking the active neuron clears the active state.
- Unchecking another neuron does not change the active neuron.
- Selecting a row in the Neuron Annotation table checks and activates the corresponding neuron.

![Neuron selection](assets/neuron_selection.png)

- Use **Q/W** to navigate to the previous or next valid neuron while preserving existing checks.
- Use **All** to check every neuron identity.
- Use **None** to clear both checked and active states.
- Enable **Show selected box labels** to display one text label for each currently rendered checked neuron. The plugin uses the `biological` value when available and otherwise displays the zero-based `neuron_id`. The text color can be changed with the **Text color** control.

## Z Layers

The **Z Layers** panel can divide the volume into half-open Z ranges. In an individual Z layer, only the corresponding Image and Labels slices and boxes whose center Z belongs to that range are shown. Checked and active identities remain global.

Bilaterally distributed or densely overlapping neurons may occlude one another in a full 3D view. Z Layers can divide the volume into two or more half-open Z ranges, making subsets of neurons easier to inspect. Each box is assigned as a whole according to its center Z coordinate. Use **Split** to create the Z layers, then use **Show** to select **All** or an individual Z layer.

![Z-layer split](assets/split.png)

![Z-layer overview](assets/z_layer.png)

Neurons outside the active Z range remain listed in gray. Their checkboxes remain operable, and checked and active states remain global. Their boxes are not rendered in the current Z-layer view, and activating one does not move the viewer outside the selected range. Q/W navigation is restricted to neurons in the active Z range.

## Neuron Annotation

The **Neuron Annotation** table contains three columns:

- `digital`: the zero-based `neuron_id` in ROI mode;
- `biological`: the biological display name;
- `annotation`: free-form annotation text.

Selecting a table row checks and activates the corresponding neuron. Excel **Load** and **Save** actions require the optional `openpyxl` dependency.

The `biological` value is also used for selected box labels. If it is empty, the plugin displays the zero-based `neuron_id`.

![Neuron annotation](assets/neuron_annotation.png)

## Labels Layer

The current launcher requires `neuron_mask.npy` because the controlled Labels layer provides the display context for label colors, opacity control, spatial metadata, and Labels slices for Z-layer views. The ROI array remains the authoritative source of neuron identity and box geometry. In ROI mode, the explicit mapping is `label_value = neuron_id + 1`.

A dense or opaque Labels display may obscure the underlying image or the Vectors ROI boxes. When the mask display is not needed, hide the Labels layer with the eye icon in napari's layer list, or set both checked and unchecked label opacity to `0`. These operations affect display only and do not modify `neuron_mask.npy`.

## Alternative installation with pip

Pixi is recommended for reproducible use. If Pixi is not available, create a Python 3.11–3.14 environment and install the plugin with napari and a Qt backend:

```bash
pip install "napari-worm-neuron-annotator[all]"
python script/launch_datasets.py
```

If the Python environment already contains a working napari and Qt installation:

```bash
pip install napari-worm-neuron-annotator
python script/launch_datasets.py
```

## Troubleshooting

### Dataset files were not found

Confirm that `DATA_DIR` points directly to the directory containing all three NPY files. Check path spelling and use a raw string for Windows paths:

```python
DATA_DIR = Path(r"D:\data\worm_dataset")
```
### Image and Labels shapes do not match

`volumes.npy` and `neuron_mask.npy` must have identical four-dimensional `(t, z, y, x)` shapes.
### napari or Qt does not start

First try the locked Pixi environment:

```bash
pixi run start
```

If the problem persists, record the terminal error, operating system, graphics hardware, and graphics-driver version when requesting support.

## Related resources
- [Plugin on PyPI](https://pypi.org/project/napari-worm-neuron-annotator/)
- [Plugin source repository](https://github.com/Wenlab/napari-worm-neuron-annotator)
- [Pixi documentation](https://pixi.sh/)
- [napari documentation](https://napari.org/)
