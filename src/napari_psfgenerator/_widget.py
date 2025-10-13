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

    # --- Parameters (merged Physical + Numerical) ---
    parameters = widgets.Container(
        widgets=[
            widgets.Label(value="<b>Parameters</b>"),
            widgets.FloatText(value=1.4, min=0, max=1.5, step=0.1, label="NA"),
            widgets.FloatText(value=632, min=0, max=1300, step=10, label="Wavelength [nm]"),
            widgets.FloatText(value=20, min=0, max=1000, step=10, label="Pixel Size [nm]"),
            widgets.FloatText(value=20, min=0, max=2000, step=10, label="Defocus Step [nm]"),
            widgets.SpinBox(value=203, label="Pixels in Pupil", min=1),
            widgets.SpinBox(value=201, label="Pixels in PSF", min=1),
            widgets.SpinBox(value=200, label="Z-Stacks", min=1),
            widgets.ComboBox(choices=["cpu", "cuda:0"], value="cpu", label="Device")
        ],
        layout="vertical"
    )

    # --- Corrections Container ---
    corrections_label = widgets.Label(value="<b>Corrections</b>")
    
    # Basic corrections
    apod_factor = widgets.CheckBox(value=False, label="Apodization Factor")
    gibson_lanni = widgets.CheckBox(value=True, label="Gibson-Lanni")
    
    # Envelope (Gaussian incident field)
    envelope = widgets.FloatText(
        value=None,
        min=0,
        max=10000,
        step=100,
        label="Envelope",
        tooltip="Size of Gaussian envelope. Leave empty for plane wave."
    )

    # Zernike Aberrations (Collapsible)
    show_zernike = widgets.CheckBox(value=False, label="▶ Zernike Aberrations")
    zernike_container = widgets.Container(
        widgets=[
            widgets.FloatText(value=0.0, min=-5.0, max=5.0, step=0.1, label="Astigmatism"),
            widgets.FloatText(value=0.0, min=-5.0, max=5.0, step=0.1, label="Defocus"),
            widgets.FloatText(value=0.0, min=-5.0, max=5.0, step=0.1, label="Coma X"),
            widgets.FloatText(value=0.0, min=-5.0, max=5.0, step=0.1, label="Coma Y"),
            widgets.FloatText(value=0.0, min=-5.0, max=5.0, step=0.1, label="Spherical"),
        ],
        layout="vertical",
        visible=False
    )

    def toggle_zernike(event):
        zernike_container.visible = show_zernike.value
        show_zernike.label = "▼ Zernike Aberrations" if show_zernike.value else "▶ Zernike Aberrations"
    
    show_zernike.changed.connect(toggle_zernike)

    # Polarization (Vectorial) (Collapsible)
    show_vectorial = widgets.CheckBox(value=False, label="▶ Polarization (Vectorial)")
    vectorial_container = widgets.Container(
        widgets=[
            widgets.FloatText(value=1.0, min=-100, max=100, step=0.1, label="e0x (Real)"),
            widgets.FloatText(value=0.0, min=-100, max=100, step=0.1, label="e0x (Imag)"),
            widgets.FloatText(value=0.0, min=-100, max=100, step=0.1, label="e0y (Real)"),
            widgets.FloatText(value=0.0, min=-100, max=100, step=0.1, label="e0y (Imag)")
        ],
        layout="vertical",
        visible=False
    )

    def toggle_vectorial(event):
        vectorial_container.visible = show_vectorial.value
        show_vectorial.label = "▼ Polarization (Vectorial)" if show_vectorial.value else "▶ Polarization (Vectorial)"
    
    show_vectorial.changed.connect(toggle_vectorial)

    # Group all corrections together
    corrections_container = widgets.Container(
        widgets=[
            corrections_label,
            apod_factor,
            gibson_lanni,
            envelope,
            show_zernike,
            zernike_container,
            show_vectorial,
            vectorial_container,
        ],
        layout="vertical"
    )

    # Buttons and Result Display
    compute_button = widgets.PushButton(text="▶ Compute and Display")
    save_button = widgets.PushButton(text="💾 Save Image")
    result_viewer = widgets.Label(value="Result will be displayed here")
    axes_button = widgets.CheckBox(value=True, label="Show XYZ Axes")

    # Define a container to hold all grouped sections
    container = widgets.Container(
        widgets=[
            propagator_type,
            parameters,
            corrections_container,
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

        # Show/hide Vectorial-specific parameters
        is_vectorial = selected_type.startswith("Vectorial")
        show_vectorial.visible = is_vectorial
        if not is_vectorial:
            show_vectorial.value = False
            vectorial_container.visible = False

        # Show/hide Zernike Astigmatism for Cartesian propagators
        is_cartesian = "Cartesian" in selected_type
        if is_cartesian and len(zernike_container) > 0:
            zernike_container[0].visible = True  # Astigmatism visible for Cartesian

    # Connect the dropdown value change to the update function
    propagator_type.changed.connect(update_propagator_params)

    # Initial update to set the correct visibility
    update_propagator_params(None)

    # Compute button callback function
    def compute_result():
        # Gather common parameters
        kwargs = {
            'n_pix_pupil': parameters[5].value,
            'n_pix_psf': parameters[6].value,
            'n_defocus': parameters[7].value,
            'device': parameters[8].value,
            'wavelength': parameters[2].value,
            'na': parameters[1].value,
            'pix_size': parameters[3].value,
            'defocus_step': parameters[4].value,
            'apod_factor': apod_factor.value,
            'gibson_lanni': gibson_lanni.value,
            'envelope': envelope.value,
            'zernike_coefficients': [
                0, 0, 
                zernike_container[2].value,  # Coma X
                zernike_container[3].value,  # Coma Y
                zernike_container[1].value,  # Defocus
                zernike_container[0].value,  # Astigmatism
                zernike_container[4].value   # Spherical
            ],
        }

        # Add specific parameters based on the propagator type
        if propagator_type.value.startswith("Scalar"):
            if propagator_type.value == "ScalarCartesian":
                propagator = ScalarCartesianPropagator(**kwargs)
            else:
                propagator = ScalarSphericalPropagator(**kwargs)
        else:
            # Combine real and imaginary parts into complex numbers
            e0x_real = vectorial_container[0].value
            e0x_imag = vectorial_container[1].value
            e0y_real = vectorial_container[2].value
            e0y_imag = vectorial_container[3].value
            
            kwargs.update({
                'e0x': complex(e0x_real, e0x_imag),
                'e0y': complex(e0y_real, e0y_imag)
            })
            if propagator_type.value == "VectorialCartesian":
                propagator = VectorialCartesianPropagator(**kwargs)
            else:
                propagator = VectorialSphericalPropagator(**kwargs)

        # Compute the field and display the result
        print(f"Computing field for {propagator_type.value}...")
        field = propagator.compute_focus_field()

        if 'Scalar' in propagator_type.value:
            field_amplitude = field.abs()
            result = (field_amplitude/field_amplitude.max()).cpu().numpy().squeeze()
        else:
            field_amplitude = ((field[:, :, :, :].abs().squeeze() ** 2).sum(dim=1)).sqrt().squeeze()
            result = (field_amplitude/field_amplitude.max()).cpu().numpy()

        # Save the computed result
        computed_result['data'] = result

        # Add image and enable 3D visualization with axes
        viewer.add_image(result, name=f"Result: {propagator_type.value}", colormap='inferno')
        viewer.axes.visible = axes_button.value
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
        dialog.setGeometry(300, 300, 600, 400)

        if dialog.exec_():
            filepath = dialog.selectedFiles()[0]
            if filepath:
                os.makedirs(os.path.dirname(filepath), exist_ok=True)
                viewer.layers[-1].save(filepath)
                result_viewer.value = f"Image saved to {filepath}"

    save_button.clicked.connect(save_computed_image)

    return container