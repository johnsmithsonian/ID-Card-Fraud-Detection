import cv2
import numpy as np
import streamlit as st
from skimage.metrics import structural_similarity as ssim
import logging
import tempfile
from results import save_collage_image

logging.basicConfig(level=logging.INFO)

st.set_page_config(page_title='ID Card Tampering Detection')

st.markdown(
    """
    <style>a
    body {
        font-family: Arial, sans-serif;
        background-color: #f4f4f4;
    }
    h1 {
        color: #333;
        text-align: center;
    }
    h2 {
        color: red;
        text-align: center;
        font-weight: bold;
    }
    .sidebar .sidebar-content {
        background-color: #f8f8f8;
        border-radius: 10px;
    }
    .stButton > button {
        background-color: #28a745;
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 5px;
        cursor: pointer;
    }
    .stButton > button:hover {
        background-color: #218838;
    }
    .stSelectbox, .stSlider {
        margin: 10px 0;
    }
    </style>
    """,
    unsafe_allow_html=True
)


def load_image(file):

    if isinstance(file, str):
        image = cv2.imread(file)
    else:
        image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

    if image is None:
        logging.error("Error loading image.")
        st.error("Error loading image. Please upload a valid image.")
    return image


def convert_to_gray(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    logging.info("Converted image to grayscale.")
    return gray


def resize_image(image, size=(600, 400)):
    resized = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    logging.info("Resized image to %s.", size)
    return resized


def calculate_ssim(image1, image2):
    ssim_index, _ = ssim(image1, image2, full=True)
    logging.info("Calculated SSIM index: %f", ssim_index)
    return ssim_index


def detect_tampering(ssim_index, threshold):
    tampering_detected = ssim_index < threshold
    if tampering_detected:
        logging.warning("Tampering detected.")
    else:
        logging.info("No tampering detected.")
    return tampering_detected


def find_image_contours(image):
    contours, _ = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    logging.info("Found %d contours in the image.", len(contours))
    return contours


def draw_contours(image, contours):
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 100:  # Filter out small contours
            cv2.drawContours(image, [contour], -1, (0, 0, 255), 2)
    logging.info("Contours drawn on image.")
    return image


def save_image(image, prefix="processed"):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg", prefix=prefix) as temp_file:
        cv2.imwrite(temp_file.name, image)
        logging.info("Saved processed image to %s.", temp_file.name)
        return temp_file.name


def display_image(image, title=None):
    if title:
        st.write(title)
    st.image(image, channels='BGR', use_column_width=True)


def display_results(ssim_index, tampering_detected, original_name):
    st.write(f'Comparing with: {original_name}')
    st.write(f'Structural Similarity Index (SSIM): {ssim_index:.2f}')
    if tampering_detected:
        st.markdown('<h2 style="color: red;">Tampering Detected.</h2>', unsafe_allow_html=True)
    else:
        st.markdown('<h2 style="color: green;">No Tampering Detected.</h2>', unsafe_allow_html=True)


def validate_image_upload(original_files, tampered):
    if len(original_files) == 0:
        st.warning("Please upload at least one original image or use the Camera.")
        return False
    if tampered is None:
        st.warning("Please upload the tampered image or use the Camera.")
        return False
    return True



def set_ssim_threshold(key):
    return st.sidebar.slider("Set SSIM Threshold", 0.0, 1.0, 0.96, key=key)  # Default to 0.96


def create_collage(original_image, tampered_image, tampering_detected):
    height = max(original_image.shape[0], tampered_image.shape[0])
    original_resized = cv2.resize(original_image,
                                  (int(original_image.shape[1] * height / original_image.shape[0]), height))
    tampered_resized = cv2.resize(tampered_image,
                                  (int(tampered_image.shape[1] * height / tampered_image.shape[0]), height))

    collage_width = original_resized.shape[1] + tampered_resized.shape[1]
    collage = np.zeros((height, collage_width, 3), dtype=np.uint8)

    collage[:original_resized.shape[0], :original_resized.shape[1]] = original_resized
    collage[:tampered_resized.shape[0], original_resized.shape[1]:] = tampered_resized

    font_scale = 1
    thickness = 2


    cv2.putText(collage, "Database Image", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness,
                cv2.LINE_AA)


    cv2.putText(collage, "Current Image", (original_resized.shape[1] + 10, 50), cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                (0, 0, 255), thickness, cv2.LINE_AA)


    text = "Tampering Detected!" if tampering_detected else "No Tampering Detected"
    cv2.putText(collage, text, (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                (0, 0, 255 if tampering_detected else 255), thickness, cv2.LINE_AA)

    logging.info("Created collage of original and tampered images.")
    return collage


def process_images(original_files, tampered, threshold):
    tampered_image = load_image(tampered)
    tampered_image_gray = resize_image(convert_to_gray(tampered_image))

    results = []
    all_tampered = True

    for original_file in original_files:
        original_image = load_image(original_file)
        reference_image_resized = resize_image(convert_to_gray(original_image))

        ssim_index = calculate_ssim(tampered_image_gray, reference_image_resized)
        tampering_detected = detect_tampering(ssim_index, threshold)
        contours = find_image_contours(tampered_image_gray)

        image_with_contours = draw_contours(tampered_image.copy(), contours)
        collage = create_collage(original_image, tampered_image, tampering_detected)

        results.append((original_image, image_with_contours, collage, ssim_index, tampering_detected))


        collage_filename = f"collage_original_{original_files.index(original_file) + 1}.jpg"
        save_collage_image(collage, collage_filename)

        if not tampering_detected:
            all_tampered = False

    return results, all_tampered


def file_upload_tab():
    st.title('Upload Images')
    st.write('Upload one or more original ID card images and a tampered image to detect tampering.')

    uploaded_originals = st.file_uploader('Upload Original Images', type=['jpg', 'jpeg', 'png'],
                                          accept_multiple_files=True)

    uploaded_tampered = st.file_uploader('Upload Tampered Image', type=['jpg', 'jpeg', 'png'])

    threshold = set_ssim_threshold(key="file_upload_threshold")

    if validate_image_upload(uploaded_originals, uploaded_tampered):
        results, all_tampered = process_images(uploaded_originals, uploaded_tampered, threshold)


        selected_index = st.selectbox("Select an original image to view results:", range(len(results)))

        original_image, processed_image, collage, ssim_index, tampering_detected = results[selected_index]

        display_results(ssim_index, tampering_detected, f"Original Image {selected_index + 1}")
        display_image(processed_image)

        collage_path = save_image(collage, prefix="collage_")
        collage_display = cv2.imread(collage_path)
        display_image(collage_display)

        st.download_button(
            label=f"Download Collage Image {selected_index + 1}",
            data=open(collage_path, "rb").read(),
            file_name=f"collage_image_{selected_index + 1}.jpg",
            mime="image/jpeg",
            key=f"download_button_{selected_index}"
        )


def webcam_tab():
    st.title('Camera Capture')
    st.write('Upload original images and capture tampered images for tampering detection.')

    uploaded_originals = st.file_uploader("Upload Original Image", type=['jpg', 'jpeg', 'png'],
                                          accept_multiple_files=False)

    tampered_image_webcam = st.camera_input("Capture Tampered Image")
    if tampered_image_webcam:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            temp_file.write(tampered_image_webcam.getbuffer())
            uploaded_tampered = temp_file.name
    else:
        uploaded_tampered = None

    threshold = set_ssim_threshold(key="webcam_threshold")

    if validate_image_upload([uploaded_originals], uploaded_tampered):
        results, all_tampered = process_images([uploaded_originals], uploaded_tampered, threshold)

        if all_tampered:
            st.markdown('<h2>The Above ID Card is Tampered.</h2>', unsafe_allow_html=True)

        original_image, processed_image, collage, ssim_index, tampering_detected = results[0]

        display_results(ssim_index, tampering_detected, "Uploaded Original Image")
        display_image(processed_image)

        collage_path = save_image(collage, prefix="collage_")
        collage_display = cv2.imread(collage_path)
        display_image(collage_display)

        st.download_button(
            label="Download Collage Image",
            data=open(collage_path, "rb").read(),
            file_name="collage_image.jpg",
            mime="image/jpeg",
            key="download_button"
        )


def main():
    st.title('ID Card Tampering Detection')

    tab1, tab2 = st.tabs(["File Upload", "Camera Capture"])

    with tab1:
        file_upload_tab()

    with tab2:
        webcam_tab()


if __name__ == '__main__':
    main()
