import os
from qtpy.QtWidgets import QFileDialog
from magicgui import widgets
from psf_generator.propagators.scalar_cartesian_propagator import ScalarCartesianPropagator
from psf_generator.propagators.scalar_spherical_propagator import ScalarSphericalPropagator
from psf_generator.propagators.vectorial_cartesian_propagator import VectorialCartesianPropagator
from psf_generator.propagators.vectorial_spherical_propagator import VectorialSphericalPropagator
from napari import current_viewer
viewer = current_viewer()  # Get the current Napari viewer

def propagators_container():
    # Dropdown for propagator type selection
    propagator_type = widgets.ComboBox(
        choices=["ScalarCartesian", "ScalarSpherical", "VectorialCartesian", "VectorialSpherical"],
        label="Select Propagator")

    # --- Physical Parameters ---
    physical_parameters = widgets.Container(
        widgets=[
            widgets.Label(value="Physical Parameters"),
            widgets.FloatText(value=1.4, min=0, max=1.5, step=0.1, label="NA"),
            widgets.FloatText(value=632, min=0, max=1300, step=10, label="Wavelength [nm]"),
            widgets.FloatText(value=20, min=0, max=1000, step=10, label="Pixel Size [nm]"),
            widgets.FloatText(value=20, min=0, max=2000, step=10, label="Defocus Step [nm]")
        ],
        layout="vertical"
    )

    device_list = ["cpu", "cuda:0"]

    # --- Numerical Parameters ---
    numerical_parameters = widgets.Container(
        widgets=[
            widgets.Label(value="Numerical Parameters"),
            widgets.SpinBox(value=203, label="Pixels in Pupil", min=1),
            widgets.SpinBox(value=201, label="Pixels in PSF", min=1),
            widgets.SpinBox(value=200, label="Z-Stacks", min=1),
                widgets.ComboBox(choices=device_list, value="cpu", label="Device")
        ],
        layout="vertical"
    )

    # --- Options ---
    options_parameters = widgets.Container(
        widgets=[
            # label on the left, widget on the right
            widgets.Label(value="Options"),
            widgets.CheckBox(value=False, label="Apodization Factor"),
            widgets.CheckBox(value=True, label="Gibson-Lanni"),
            widgets.FloatText(value=0.0, min=0, max=5.0, step=0.1, label="Zernike Astigmatism"),
            widgets.FloatText(value=0.0, min=0, max=5.0, step=0.1, label="Zernike Defocus"),
            widgets.FloatText(value=1.0, min=0, max=100, step=0.1, label="e0x"),
            widgets.FloatText(value=0.0, min=0, max=100, step=0.1, label="e0y")
        ],
        layout="vertical",
    )

    # Buttons and Result Display
    compute_button = widgets.PushButton(text="Compute and Display")
    save_button = widgets.PushButton(text="Save Image")
    result_viewer = widgets.Label(value="Result will be displayed here")
    axes_button = widgets.CheckBox(value=True, label="Show XYZ Axes")

    # Define a container to hold all grouped sections
    container = widgets.Container(
        widgets=[
            propagator_type,
            physical_parameters,
            numerical_parameters,
            options_parameters,
            compute_button,
            save_button,
            result_viewer,
            axes_button
        ],
        layout="vertical"
    )

    # Store the computed result for saving
    computed_result = {'data': None}

    # Function to update visible widgets based on the selected propagator type
    def update_propagator_params(event):
        selected_type = propagator_type.value

        # Show/hide Vectorial-specific parameters in Options
        is_vectorial = selected_type.startswith("Vectorial")
        options_parameters[5].visible = is_vectorial  # e0x
        options_parameters[6].visible = is_vectorial  # e0y

        # Show/hide Zernike Astigmatism for spherical propagators
        is_spherical = "Cartesian" in selected_type
        options_parameters[3].visible = is_spherical  # Zernike Astigmatism

    # Connect the dropdown value change to the update function
    propagator_type.changed.connect(update_propagator_params)

    # Initial update to set the correct visibility
    update_propagator_params(None)

    # Store the computed result for saving
    computed_result = {'data': None}  # Dictionary to hold the result image data

    # Compute button callback function
    def compute_result():
        # Gather common parameters
        kwargs = {
            'n_pix_pupil': numerical_parameters[1].value,
            'n_pix_psf': numerical_parameters[2].value,
            'n_defocus': numerical_parameters[3].value,
            'device': numerical_parameters[4].value,
            'wavelength': physical_parameters[2].value,
            'na': physical_parameters[1].value,
            'pix_size': physical_parameters[3].value,
            'defocus_step': physical_parameters[4].value,
            'apod_factor': options_parameters[1].value,
            'gibson_lanni': options_parameters[2].value,
            'zernike_coefficients': [0, 0, 0, 0, options_parameters[4].value, options_parameters[3].value],
        }

        # Add specific parameters based on the propagator type
        if propagator_type.value.startswith("Scalar"):
            if propagator_type.value == "ScalarCartesian":
                propagator = ScalarCartesianPropagator(**kwargs)
            else:
                propagator = ScalarSphericalPropagator(**kwargs)
        else:
            kwargs.update({
                'e0x': options_parameters[5].value,
                'e0y': options_parameters[6].value
            })
            if propagator_type.value == "VectorialCartesian":
                propagator = VectorialCartesianPropagator(**kwargs)
            else:
                propagator = VectorialSphericalPropagator(**kwargs)

        # Compute the field and display the result
        print(f"Computing field for {propagator_type.value}...")
        field = propagator.compute_focus_field()

        if 'Scalar' in propagator_type.value:
            # field_amplitude = torch.abs(field)
            field_amplitude = field.abs()
            result = (field_amplitude/field_amplitude.max()).cpu().numpy().squeeze()
        else:
            # field_amplitude = torch.sqrt(torch.sum(torch.abs(field[:, :, :, :].squeeze()) ** 2, dim=1)).squeeze()
            field_amplitude = ((field[:, :, :, :].abs().squeeze() ** 2).sum(dim=1)).sqrt().squeeze()
            result = (field_amplitude/field_amplitude.max()).cpu().numpy()

        # Save the computed result
        computed_result['data'] = result

        # Add image and enable 3D visualization with axes
        viewer.add_image(result, name=f"Result: {propagator_type.value}", colormap='inferno')
        viewer.axes.visible = axes_button.value  # Show XYZ axes
        viewer.axes.colored = False
        viewer.dims.axis_labels = ["z", "y", "x"]
        result_viewer.value = f"Computation complete! Shape: {result.shape}"


    # Connect the compute button to the compute function
    compute_button.clicked.connect(compute_result)

    # Save button callback function
    def save_computed_image():
        if computed_result['data'] is None:
            result_viewer.value = "No image to save. Please compute an image first."
            return

        # Open a file save dialog
        dialog = QFileDialog()
        dialog.setAcceptMode(QFileDialog.AcceptSave)
        dialog.setNameFilters(["TIFF files (*.tif)", "All files (*)"])
        dialog.setDefaultSuffix("tif")
        dialog.setWindowTitle("Save Image")
        dialog.setGeometry(300, 300, 600, 400)  # Set dialog position and size (x, y, width, height)

        if dialog.exec_():
            filepath = dialog.selectedFiles()[0]
            if filepath:
                os.makedirs(os.path.dirname(filepath), exist_ok=True)  # Ensure directory exists
                viewer.layers[-1].save(filepath)
                result_viewer.value = f"Image saved to {filepath}"


    save_button.clicked.connect(save_computed_image)

    return container
