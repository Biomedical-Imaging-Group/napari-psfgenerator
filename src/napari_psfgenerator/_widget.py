import os
import json
from qtpy.QtWidgets import QFileDialog, QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QWidget, QFormLayout
from magicgui import widgets
from psf_generator.propagators.scalar_cartesian_propagator import ScalarCartesianPropagator
from psf_generator.propagators.scalar_spherical_propagator import ScalarSphericalPropagator
from psf_generator.propagators.vectorial_cartesian_propagator import VectorialCartesianPropagator
from psf_generator.propagators.vectorial_spherical_propagator import VectorialSphericalPropagator
from napari import current_viewer

# Check if CUDA is available and count GPUs
try:
    import torch
    CUDA_AVAILABLE = torch.cuda.is_available()
    NUM_GPUS = torch.cuda.device_count() if CUDA_AVAILABLE else 0
except ImportError:
    CUDA_AVAILABLE = False
    NUM_GPUS = 0

viewer = current_viewer()  # Get the current Napari viewer

def propagators_container():
    # Dropdown for propagator type selection
    propagator_type = widgets.ComboBox(
        choices=["ScalarCartesian", "ScalarSpherical", "VectorialCartesian", "VectorialSpherical"],
        label="Select Propagator")

    # Polarization (Vectorial) - Button interface (defined early so it can be added to parameters)
    # Store polarization values
    polarization_params = {
        'e0x_real': 1.0,
        'e0x_imag': 0.0,
        'e0y_real': 0.0,
        'e0y_imag': 0.0
    }

    polarization_button = widgets.PushButton(text="Incident polarization", visible=False)

    def open_polarization_dialog():
        dialog = QDialog()
        dialog.setWindowTitle("Polarization (Vectorial)")
        dialog.setMinimumWidth(400)

        main_layout = QVBoxLayout()
        form_layout = QFormLayout()

        # Create parameter widgets
        param_widgets = {}
        param_data = [
            ('e0x_real', 'E_x (Real):', polarization_params['e0x_real'], "Real part of x-component of electric field"),
            ('e0x_imag', '    (Imag):', polarization_params['e0x_imag'], "Imaginary part of x-component of electric field"),
            ('e0y_real', 'E_y (Real):', polarization_params['e0y_real'], "Real part of y-component of electric field"),
            ('e0y_imag', '    (Imag):', polarization_params['e0y_imag'], "Imaginary part of y-component of electric field"),
        ]

        for key, label, value, tooltip in param_data:
            widget = widgets.FloatText(value=value, min=-100, max=100, step=0.1, tooltip=tooltip)
            param_widgets[key] = widget
            form_layout.addRow(label, widget.native)

        main_layout.addLayout(form_layout)

        # Button layout
        button_layout = QHBoxLayout()
        save_btn = QPushButton("Save")
        reset_btn = QPushButton("Reset")
        button_layout.addWidget(save_btn)
        button_layout.addWidget(reset_btn)

        def save_params():
            for key, widget in param_widgets.items():
                polarization_params[key] = widget.value
            dialog.accept()

        def reset_params():
            defaults = {'e0x_real': 1.0, 'e0x_imag': 0.0, 'e0y_real': 0.0, 'e0y_imag': 0.0}
            for key, widget in param_widgets.items():
                widget.value = defaults[key]

        save_btn.clicked.connect(save_params)
        reset_btn.clicked.connect(reset_params)

        main_layout.addLayout(button_layout)
        dialog.setLayout(main_layout)
        dialog.exec_()

    polarization_button.clicked.connect(open_polarization_dialog)

    # Store advanced parameters
    advanced_params = {
        'pix_size': 20.0,
        'defocus_step': 20.0,
        'n_pix_pupil': 203,
        'n_pix_psf': 201,
        'n_defocus': 200,
        'device': 'cpu'
    }

    advanced_params_button = widgets.PushButton(text="More parameters")

    def open_advanced_params_dialog():
        dialog = QDialog()
        dialog.setWindowTitle("Advanced Parameters")
        dialog.setMinimumWidth(400)

        main_layout = QVBoxLayout()
        form_layout = QFormLayout()

        # Create parameter widgets
        param_widgets = {}
        param_data = [
            ('pix_size', 'Pixel Size [nm]:', advanced_params['pix_size'], 0, 1000, 10, "Pixel size of the PSF"),
            ('defocus_step', 'Defocus Step [nm]:', advanced_params['defocus_step'], 0, 2000, 10, "Step size between z-planes"),
            ('n_pix_pupil', 'Pixels in Pupil:', advanced_params['n_pix_pupil'], 1, 1000, 1, "Number of pixels used to sample the pupil plane"),
            ('n_pix_psf', 'Pixels in PSF:', advanced_params['n_pix_psf'], 1, 1000, 1, "Number of pixels in the output PSF image (x and y)"),
            ('n_defocus', 'Number of Z planes:', advanced_params['n_defocus'], 1, 1000, 1, "Number of z-planes for the output PSF"),
        ]

        for key, label, value, min_val, max_val, step, tooltip in param_data:
            if key in ['n_pix_pupil', 'n_pix_psf', 'n_defocus']:
                widget = widgets.SpinBox(value=int(value), min=min_val, max=max_val, tooltip=tooltip)
            else:
                widget = widgets.FloatText(value=value, min=min_val, max=max_val, step=step, tooltip=tooltip)
            param_widgets[key] = widget
            form_layout.addRow(label, widget.native)

        # Device selection
        device_choices = ["cpu"]
        for i in range(NUM_GPUS):
            device_choices.append(f"cuda:{i}")
        device_widget = widgets.ComboBox(choices=device_choices, value=advanced_params['device'],
                                        label="Device", tooltip="Computation device (CPU or CUDA GPU)")
        param_widgets['device'] = device_widget
        form_layout.addRow("Device:", device_widget.native)

        main_layout.addLayout(form_layout)

        # Button layout
        button_layout = QHBoxLayout()
        save_btn = QPushButton("Save")
        reset_btn = QPushButton("Reset")
        button_layout.addWidget(save_btn)
        button_layout.addWidget(reset_btn)

        def save_params():
            for key, widget in param_widgets.items():
                advanced_params[key] = widget.value
            dialog.accept()

        def reset_params():
            defaults = {'pix_size': 20.0, 'defocus_step': 20.0, 'n_pix_pupil': 203,
                       'n_pix_psf': 201, 'n_defocus': 200, 'device': 'cpu'}
            for key, widget in param_widgets.items():
                widget.value = defaults[key]

        save_btn.clicked.connect(save_params)
        reset_btn.clicked.connect(reset_params)

        main_layout.addLayout(button_layout)
        dialog.setLayout(main_layout)
        dialog.exec_()

    advanced_params_button.clicked.connect(open_advanced_params_dialog)

    # --- Parameters ---
    parameters = widgets.Container(
        widgets=[
            widgets.Label(value="<b>Parameters</b>"),
            widgets.FloatText(value=1.4, min=0, max=1.5, step=0.1, label="NA",
                            tooltip="Numerical aperture of the objective lens"),
            widgets.FloatText(value=632, min=0, max=1300, step=10, label="Wavelength [nm]",
                            tooltip="Wavelength of incident light"),
            advanced_params_button,
            polarization_button
        ],
        layout="vertical"
    )

    # --- Corrections Container ---
    corrections_label = widgets.Label(value="<b>Corrections</b>")

    # Apodization Factor
    apod_factor = widgets.CheckBox(
        value=False,
        label="Apodization Factor",
        tooltip="Apply apodization factor sqrt(cos(theta))"
    )

    # Envelope (Gaussian incident field)
    envelope_enabled = widgets.CheckBox(
        value=False,
        label="Envelope",
        tooltip="Gaussian envelope factor for incident field (unitless, order 1)"
    )
    envelope = widgets.FloatText(
        value=1.0,
        min=0.1,
        max=2.0,
        step=0.1,
        label="Envelope Factor",
        visible=False
    )

    # Gibson-Lanni Correction with Popup
    gibson_lanni = widgets.CheckBox(
        value=False,
        label="Gibson-Lanni",
        tooltip="Apply Gibson-Lanni correction for stratified media"
    )

    # Store Gibson-Lanni parameters (thicknesses stored in nm internally)
    gibson_lanni_params = {
        'z_p': 1000.0,
        'n_s': 1.3,
        'n_g': 1.5,
        'n_g0': 1.5,
        't_g': 170000.0,
        't_g0': 170000.0,
        'n_i': 1.5,
        'n_i0': 1.5,
        't_i0': 100000.0
    }

    gibson_lanni_button = widgets.PushButton(text="Parameters", visible=False)

    def open_gibson_lanni_dialog():
        dialog = QDialog()
        dialog.setWindowTitle("Gibson-Lanni Parameters")
        dialog.setMinimumWidth(400)

        main_layout = QVBoxLayout()
        form_layout = QFormLayout()

        # Create parameter widgets with labels (thicknesses in microns for UI)
        param_widgets = {}
        param_data = [
            ('z_p', 'z_p [nm]:', gibson_lanni_params['z_p'], 0, 10000, 100, 1.0, "Depth of focal plane in sample"),
            ('n_s', 'n_s:', gibson_lanni_params['n_s'], 1.0, 2.0, 0.01, 1.0, "Refractive index of sample"),
            ('n_g', 'n_g:', gibson_lanni_params['n_g'], 1.0, 2.0, 0.01, 1.0, "Refractive index of cover slip"),
            ('n_g0', 'n_g0:', gibson_lanni_params['n_g0'], 1.0, 2.0, 0.01, 1.0, "Design refractive index of cover slip"),
            ('t_g', 't_g [μm]:', gibson_lanni_params['t_g'], 0, 500, 1, 1000.0, "Thickness of cover slip"),
            ('t_g0', 't_g0 [μm]:', gibson_lanni_params['t_g0'], 0, 500, 1, 1000.0, "Design thickness of cover slip"),
            ('n_i', 'n_i:', gibson_lanni_params['n_i'], 1.0, 2.0, 0.01, 1.0, "Refractive index of immersion medium"),
            ('n_i0', 'n_i0:', gibson_lanni_params['n_i0'], 1.0, 2.0, 0.01, 1.0, "Design refractive index of immersion medium"),
            ('t_i0', 't_i0 [μm]:', gibson_lanni_params['t_i0'], 0, 500, 1, 1000.0, "Design thickness of immersion medium"),
        ]

        for key, label, value, min_val, max_val, step, conversion, tooltip in param_data:
            display_value = value / conversion
            widget = widgets.FloatText(value=display_value, min=min_val, max=max_val, step=step, tooltip=tooltip)
            param_widgets[key] = (widget, conversion)
            form_layout.addRow(label, widget.native)

        main_layout.addLayout(form_layout)

        # Button layout
        button_layout = QHBoxLayout()
        save_btn = QPushButton("Save")
        reset_btn = QPushButton("Reset")
        button_layout.addWidget(save_btn)
        button_layout.addWidget(reset_btn)

        def save_params():
            for key, (widget, conversion) in param_widgets.items():
                gibson_lanni_params[key] = widget.value * conversion
            dialog.accept()

        def reset_params():
            defaults = {'z_p': 1000.0, 'n_s': 1.3, 'n_g': 1.5, 'n_g0': 1.5,
                       't_g': 170000.0, 't_g0': 170000.0, 'n_i': 1.5, 'n_i0': 1.5, 't_i0': 100000.0}
            for key, (widget, conversion) in param_widgets.items():
                widget.value = defaults[key] / conversion

        save_btn.clicked.connect(save_params)
        reset_btn.clicked.connect(reset_params)

        main_layout.addLayout(button_layout)
        dialog.setLayout(main_layout)
        dialog.exec_()

    gibson_lanni_button.clicked.connect(open_gibson_lanni_dialog)

    # Zernike Aberrations with Popup
    show_zernike = widgets.CheckBox(
        value=False,
        label="Zernike Aberrations",
        tooltip="Add Zernike aberration coefficients"
    )

    # Store Zernike parameters
    zernike_params = {
        'astigmatism': 0.0,
        'defocus': 0.0,
        'coma_x': 0.0,
        'coma_y': 0.0,
        'spherical': 0.0
    }

    zernike_button = widgets.PushButton(text="Parameters", visible=False)

    def open_zernike_dialog():
        dialog = QDialog()
        dialog.setWindowTitle("Zernike Aberrations")
        dialog.setMinimumWidth(400)

        main_layout = QVBoxLayout()
        form_layout = QFormLayout()

        # Check if using spherical propagator
        is_spherical = "Spherical" in propagator_type.value

        # Create parameter widgets with labels
        param_widgets = {}
        param_data = [
            ('astigmatism', 'Astigmatism:', zernike_params['astigmatism'], "Zernike astigmatism coefficient", False),
            ('defocus', 'Defocus:', zernike_params['defocus'], "Zernike defocus coefficient", True),
            ('coma_x', 'Coma X:', zernike_params['coma_x'], "Zernike coma X coefficient", False),
            ('coma_y', 'Coma Y:', zernike_params['coma_y'], "Zernike coma Y coefficient", False),
            ('spherical', 'Spherical:', zernike_params['spherical'], "Zernike spherical aberration coefficient", True),
        ]

        for key, label, value, tooltip, allowed_for_spherical in param_data:
            widget = widgets.FloatText(value=value, min=-5.0, max=5.0, step=0.1, tooltip=tooltip)

            # Disable widget if using spherical propagator and this param is not allowed
            if is_spherical and not allowed_for_spherical:
                widget.enabled = False
                widget.native.setStyleSheet("QDoubleSpinBox { color: gray; }")

            param_widgets[key] = widget
            form_layout.addRow(label, widget.native)

        main_layout.addLayout(form_layout)

        # Button layout
        button_layout = QHBoxLayout()
        save_btn = QPushButton("Save")
        reset_btn = QPushButton("Reset")
        button_layout.addWidget(save_btn)
        button_layout.addWidget(reset_btn)

        def save_params():
            for key, widget in param_widgets.items():
                zernike_params[key] = widget.value
            dialog.accept()

        def reset_params():
            for widget in param_widgets.values():
                widget.value = 0.0

        save_btn.clicked.connect(save_params)
        reset_btn.clicked.connect(reset_params)

        main_layout.addLayout(button_layout)
        dialog.setLayout(main_layout)
        dialog.exec_()

    zernike_button.clicked.connect(open_zernike_dialog)

    def toggle_envelope(event):
        envelope.visible = envelope_enabled.value

    envelope_enabled.changed.connect(toggle_envelope)

    def toggle_gibson_lanni(event):
        gibson_lanni_button.visible = gibson_lanni.value

    gibson_lanni.changed.connect(toggle_gibson_lanni)

    def toggle_zernike(event):
        zernike_button.visible = show_zernike.value

    show_zernike.changed.connect(toggle_zernike)

    # Group all corrections together
    corrections_container = widgets.Container(
        widgets=[
            corrections_label,
            apod_factor,
            envelope_enabled,
            envelope,
            gibson_lanni,
            gibson_lanni_button,
            show_zernike,
            zernike_button,
        ],
        layout="vertical"
    )

    # Action buttons
    compute_button = widgets.PushButton(text="▶ Compute and Display")
    result_viewer = widgets.Label(value="")
    axes_button = widgets.CheckBox(value=True, label="Show XYZ Axes")
    save_button = widgets.PushButton(text="💾 Save Image")
    export_params_button = widgets.PushButton(text="📤 Export Parameters")
    load_params_button = widgets.PushButton(text="📥 Load Parameters")

    # Define a container to hold all grouped sections
    container = widgets.Container(
        widgets=[
            propagator_type,
            parameters,
            corrections_container,
            compute_button,
            result_viewer,
            axes_button,
            save_button,
            export_params_button,
            load_params_button
        ],
        layout="vertical"
    )

    # Add stretch to push action buttons to the bottom
    # Insert stretch before the action buttons
    layout = container.native.layout()
    layout.insertStretch(3, 1)  # Insert stretch after corrections_container (index 2)

    # Store the computed result for saving
    computed_result = {'data': None}

    # Function to update visible widgets based on the selected propagator type
    def update_propagator_params(event):
        selected_type = propagator_type.value

        # Show/hide Polarization button for Vectorial propagators
        is_vectorial = selected_type.startswith("Vectorial")
        polarization_button.visible = is_vectorial

    # Connect the dropdown value change to the update function
    propagator_type.changed.connect(update_propagator_params)

    # Initial update to set the correct visibility
    update_propagator_params(None)

    # Compute button callback function
    def compute_result():
        # Gather common parameters
        kwargs = {
            'n_pix_pupil': advanced_params['n_pix_pupil'],
            'n_pix_psf': advanced_params['n_pix_psf'],
            'n_defocus': advanced_params['n_defocus'],
            'device': advanced_params['device'],
            'wavelength': parameters[2].value,
            'na': parameters[1].value,
            'pix_size': advanced_params['pix_size'],
            'defocus_step': advanced_params['defocus_step'],
            'apod_factor': apod_factor.value,
            'gibson_lanni': gibson_lanni.value,
            'envelope': envelope.value if envelope_enabled.value else None,
            'zernike_coefficients': [
                0, 0,
                zernike_params['coma_x'],
                zernike_params['coma_y'],
                zernike_params['defocus'],
                zernike_params['astigmatism'],
                zernike_params['spherical']
            ],
        }

        # Add Gibson-Lanni parameters if enabled
        if gibson_lanni.value:
            kwargs.update(gibson_lanni_params)

        # Add specific parameters based on the propagator type
        if propagator_type.value.startswith("Scalar"):
            if propagator_type.value == "ScalarCartesian":
                propagator = ScalarCartesianPropagator(**kwargs)
            else:
                propagator = ScalarSphericalPropagator(**kwargs)
        else:
            # Combine real and imaginary parts into complex numbers
            kwargs.update({
                'e0x': complex(polarization_params['e0x_real'], polarization_params['e0x_imag']),
                'e0y': complex(polarization_params['e0y_real'], polarization_params['e0y_imag'])
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

    # Export parameters to JSON
    def export_parameters():
        metadata = {
            'propagator_type': propagator_type.value,
            'parameters': {
                'na': parameters[1].value,
                'wavelength': parameters[2].value,
                'pix_size': advanced_params['pix_size'],
                'defocus_step': advanced_params['defocus_step'],
                'n_pix_pupil': advanced_params['n_pix_pupil'],
                'n_pix_psf': advanced_params['n_pix_psf'],
                'n_defocus': advanced_params['n_defocus'],
                'device': advanced_params['device'],
            },
            'corrections': {
                'apod_factor': apod_factor.value,
                'envelope_enabled': envelope_enabled.value,
                'envelope': envelope.value,
                'gibson_lanni': gibson_lanni.value,
                'gibson_lanni_params': gibson_lanni_params.copy(),
                'zernike_enabled': show_zernike.value,
                'zernike_params': zernike_params.copy(),
            },
            'polarization': polarization_params.copy()
        }

        dialog = QFileDialog()
        dialog.setAcceptMode(QFileDialog.AcceptSave)
        dialog.setNameFilters(["JSON files (*.json)", "All files (*)"])
        dialog.setDefaultSuffix("json")
        dialog.setWindowTitle("Export Parameters")
        dialog.setGeometry(300, 300, 600, 400)

        if dialog.exec_():
            filepath = dialog.selectedFiles()[0]
            if filepath:
                with open(filepath, 'w') as f:
                    json.dump(metadata, f, indent=2)
                result_viewer.value = f"Parameters exported to {filepath}"

    export_params_button.clicked.connect(export_parameters)

    # Load parameters from JSON
    def load_parameters():
        dialog = QFileDialog()
        dialog.setAcceptMode(QFileDialog.AcceptOpen)
        dialog.setNameFilters(["JSON files (*.json)", "All files (*)"])
        dialog.setWindowTitle("Load Parameters")
        dialog.setGeometry(300, 300, 600, 400)

        if dialog.exec_():
            filepath = dialog.selectedFiles()[0]
            if filepath:
                try:
                    with open(filepath, 'r') as f:
                        metadata = json.load(f)

                    # Load propagator type
                    propagator_type.value = metadata.get('propagator_type', 'ScalarCartesian')

                    # Load parameters
                    params = metadata.get('parameters', {})
                    parameters[1].value = params.get('na', 1.4)
                    parameters[2].value = params.get('wavelength', 632)
                    advanced_params['pix_size'] = params.get('pix_size', 20)
                    advanced_params['defocus_step'] = params.get('defocus_step', 20)
                    advanced_params['n_pix_pupil'] = params.get('n_pix_pupil', 203)
                    advanced_params['n_pix_psf'] = params.get('n_pix_psf', 201)
                    advanced_params['n_defocus'] = params.get('n_defocus', 200)
                    advanced_params['device'] = params.get('device', 'cpu')

                    # Load corrections
                    corrections = metadata.get('corrections', {})
                    apod_factor.value = corrections.get('apod_factor', False)
                    envelope_enabled.value = corrections.get('envelope_enabled', False)
                    envelope.value = corrections.get('envelope', 1.0)
                    gibson_lanni.value = corrections.get('gibson_lanni', False)

                    # Load Gibson-Lanni params
                    gl_params = corrections.get('gibson_lanni_params', {})
                    gibson_lanni_params.update(gl_params)

                    # Load Zernike params
                    show_zernike.value = corrections.get('zernike_enabled', False)
                    z_params = corrections.get('zernike_params', {})
                    zernike_params.update(z_params)

                    # Load polarization
                    polarization = metadata.get('polarization', {})
                    polarization_params.update(polarization)

                    # Update visibility based on loaded values
                    update_propagator_params(None)

                    result_viewer.value = f"Parameters loaded from {filepath}"
                except Exception as e:
                    result_viewer.value = f"Error loading parameters: {str(e)}"

    load_params_button.clicked.connect(load_parameters)

    return container