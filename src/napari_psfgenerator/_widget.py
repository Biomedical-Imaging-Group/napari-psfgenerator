from magicgui import widgets
import torch
import numpy as np
from propagators import ScalarCartesianPropagator, VectorialCartesianPropagator, ScalarPolarPropagator, VectorialPolarPropagator
from napari import current_viewer

viewer = current_viewer()  # Get the current Napari viewer

def propagators_container():
    # Dropdown for propagator type selection
    propagator_type = widgets.ComboBox(
        choices=["ScalarCartesian", "ScalarPolar", "VectorialCartesian", "VectorialPolar"],
        label="Select Propagator")

    # --- Physical Parameters ---
    physical_parameters = widgets.Container(
        widgets=[
            widgets.Label(value="Physical Parameters"),
            widgets.FloatText(value=1.4, min=0, max=1.5, step=0.1, label="NA"),
            widgets.FloatText(value=632, min=0, max=1300, step=10, label="Wavelength [nm]"),
            widgets.FloatText(value=2000, min=0, max=5000, step=100, label="Lateral FOV [nm]"),
            widgets.FloatText(value=4000, min=0, max=5000, step=100, label="Axial FOV (Defocus) [nm]")
        ],
        layout="vertical"
    )

    # --- Numerical Parameters ---
    numerical_parameters = widgets.Container(
        widgets=[
            widgets.Label(value="Numerical Parameters"),
            widgets.SpinBox(value=203, label="Pixels in Pupil", min=1),
            widgets.SpinBox(value=201, label="Pixels in PSF", min=1),
            widgets.SpinBox(value=200, label="Z-Stacks", min=1)
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

    # Button to trigger computation
    compute_button = widgets.PushButton(text="Compute")

    # Result display
    result_viewer = widgets.Label(value="Result will be displayed here")

    # Define a container to hold all grouped sections
    container = widgets.Container(
        widgets=[
            propagator_type,
            physical_parameters,
            numerical_parameters,
            options_parameters,
            compute_button,
            result_viewer
        ],
        layout="vertical"
    )

    # Function to update visible widgets based on the selected propagator type
    def update_propagator_params(event):
        selected_type = propagator_type.value

        # Show/hide Vectorial-specific parameters in Options
        options_parameters[5].visible = selected_type.startswith("Vectorial")
        options_parameters[6].visible = selected_type.startswith("Vectorial")

    # Connect the dropdown value change to the update function
    propagator_type.changed.connect(update_propagator_params)

    # Initial update to set the correct visibility
    update_propagator_params(None)

    # Compute button callback function
    def compute_result():
        # Gather common parameters
        kwargs = {
            'n_pix_pupil': numerical_parameters[1].value,
            'n_pix_psf': numerical_parameters[2].value,
            'n_defocus': numerical_parameters[3].value,
            'wavelength': physical_parameters[2].value,
            'na': physical_parameters[1].value,
            'fov': physical_parameters[3].value,
            'defocus_min': -physical_parameters[4].value,
            'defocus_max': physical_parameters[4].value,
            'apod_factor': options_parameters[1].value,
            'gibson_lanni': options_parameters[2].value,
            'zernike_coefficients': [0, 0, 0, options_parameters[3].value,options_parameters[4].value],
        }

        # Add specific parameters based on the propagator type
        if propagator_type.value.startswith("Scalar"):
            if propagator_type.value == "ScalarCartesian":
                propagator = ScalarCartesianPropagator(**kwargs)
            else:
                propagator = ScalarPolarPropagator(**kwargs)
        else:
            kwargs.update({
                'e0x': options_parameters[5].value,
                'e0y': options_parameters[6].value
            })
            if propagator_type.value == "VectorialCartesian":
                propagator = VectorialCartesianPropagator(**kwargs)
            else:
                propagator = VectorialPolarPropagator(**kwargs)

        # Compute the field and display the result
        print(f"Computing field for {propagator_type.value}...")
        field = propagator.compute_focus_field()

        if 'Scalar' in propagator_type.value:
            field_amplitude = torch.abs(field)
            result = (field_amplitude/field_amplitude.max()).numpy().squeeze()
        else:
            field_amplitude = torch.sqrt(torch.sum(torch.abs(field[:, :, :, :].squeeze()) ** 2, dim=1)).squeeze()
            result = (field_amplitude/field_amplitude.max()).numpy()

        viewer.add_image(result, name=f"Result: {propagator_type.value}", colormap='inferno')
        result_viewer.value = f"Computation complete! Shape: {result.shape}"

    # Connect the compute button to the compute function
    compute_button.clicked.connect(compute_result)

    return container
